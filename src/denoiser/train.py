from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Literal, TYPE_CHECKING

import numpy as np
import torch
import torch.utils.data
from torch import optim

from denoiser.data.data_loader import PairdDataset
from denoiser.data.data_transformations import (
    compose_transformations,
    load_img_clean,
    paring_clean_noisy,
    random_crop,
    standardize_img,
)
from denoiser.models.unet import UNet
from denoiser.utils.get_logger import create_logger
from denoiser.utils.tensorboard_log import TensorBoard
from denoiser.utils.trainer import TrainTrainer

if TYPE_CHECKING:
    from denoiser.configs.config import TrainConfig


def train(
    train_data_path: Path,
    train_config: TrainConfig,
    valid_data_path: Path | None = None,
    limit: int | None = None,
    verbose: Literal["debug", "info", "error"] = "info",
) -> None:
    """Run training for denoiser.

    Args:
        train_data_path: Path to training data directory
        valid_data_path: Path to validation data directory
        train_config: Training configuration
        limit: Optional limit on number of samples to use
        verbose: Logging verbosity level

    Raises:
        FileNotFoundError: If pretrained model file is specified but not found
    """
    seed = 42
    random.seed(seed)
    # Set up modern numpy random generator for reproducibility
    _ = np.random.default_rng(seed)  # Ensures reproducible numpy operations
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_dir = Path(train_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / train_config.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger("denoiser", log_dir, verbose)
    logger.info("Created output and log directories.")

    device = train_config.device

    # choufuku
    if pretrain_model_path := train_config.pretrain_model_path:
        pretrain_model_path = Path(pretrain_model_path)
        if not pretrain_model_path.is_file():
            msg = f"Pretrained model file not found: {pretrain_model_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info(f"Using pretrained model from: {pretrain_model_path}")

    logger.info("Data loading and preprocessing...")

    paring_words = train_config.pairing_keywords  # Optional[PairingKeyWards]
    noise_sigma = train_config.noise_sigma
    crop_size = train_config.cropsize  # Default 256x256 crop

    clean_img_loader = load_img_clean(paring_words)
    noisy_img_loader = paring_clean_noisy(paring_words, noise_sigma)

    # Create augmentation functions
    crop_fn = random_crop(crop_size)
    augmentation_fn = compose_transformations([crop_fn])

    logger.info(f"Clean image loader configured: {clean_img_loader.__name__}")
    logger.info(f"Noisy image loader configured: {noisy_img_loader.__name__}")
    logger.info(f"Noise sigma: {noise_sigma}")
    logger.info(f"Crop size: {crop_size}")
    logger.info(f"Augmentation pipeline configured with {len([crop_fn])} transforms")

    # Create standardization function (normalize to [0, 1] range)
    # For RGB images: convert uint8 [0, 255] to float32 [0, 1]
    standardization_fn = standardize_img(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))

    dataset = PairdDataset(
        train_data_path,
        data_loading_fn=clean_img_loader,
        img_standardization_fn=standardization_fn,
        paring_fn=noisy_img_loader,
        data_augmentation_fn=augmentation_fn,
        img_read_keywards=paring_words,
        noise_sigma=noise_sigma,
        limit=limit,
    )
    dataset_size = len(dataset)
    logger.info(f"Dataset size: {dataset_size} samples")
    train_size = int(0.8 * dataset_size)
    valid_size = dataset_size - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    logger.info(f"Train/Validation split: {train_size}/{valid_size} samples")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    logger.info("Data loaders created.")

    # Build models...
    models = UNet(in_ch=3, out_ch=3, base_ch=64).to(device)

    if not pretrain_model_path:
        logger.info("No pretrained model specified, initializing new model.")
    else:
        logger.info(f"Loading pretrained model from {pretrain_model_path}")
        checkpoint = torch.load(pretrain_model_path, map_location=device)
        models.load_state_dict(checkpoint["model_state_dict"])

    optimizer = optim.Adam(models.parameters(), lr=train_config.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    input_shape = next(iter(val_loader))[0].shape  # Assuming (batch_size, channels, height, width)
    logger.info("Start training ...")
    logger.info(f"Input shape: {input_shape}")

    trainer = TrainTrainer(
        models,
        optimizer,
        train_config,
        train_loader,
        val_loader,
    )

    tb_logger = TensorBoard(log_dir=output_dir / "tensorboard", enabled=train_config.tensorboard)
    if train_config.tensorboard:
        logger.info(f"TensorBoard logging enabled: {output_dir / 'tensorboard'}")

    if tb_logger.enabled:
        sample_input = torch.randn(1, 3, train_config.cropsize, train_config.cropsize).to(device)
        tb_logger.log_model_graph(models, sample_input)
        logger.info("Model graph logged to TensorBoard.")

    train_start_time = time.time()

    iteration = 0
    best_val_loss = float("inf")

    while iteration < train_config.iteration:
        iteration += 1
        train_loss = trainer.train_step()

        if iteration % train_config.interval == 0 or iteration == train_config.iteration:
            models.eval()
            val_losses = trainer.val_step()
            val_loss = val_losses["Loss"]

            # Extract train loss value (it's a dict now)
            train_loss_value = train_loss["Loss"] if isinstance(train_loss, dict) else train_loss

            logger.info(f"Iteration {iteration}: Train Loss={train_loss_value:.4f}, Val Loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trainer.save_model(output_dir / "best_model.pth")
                logger.info(f"New best model saved at iteration {iteration}")

            tb_logger.log_training_metrics(
                train_loss=train_loss_value, val_loss=val_loss, learning_rate=scheduler.get_last_lr()[0], step=iteration
            )

            tb_logger.log_sample_predictions(
                clean_images=clean_samples,
                noisy_images=noisy_samples,
                predictions=pred_samples,
                step=iteration,
                max_samples=n_samples,
            )

            models.train()
            scheduler.step()

    total_time = time.time()
    logger.info(f"Training completed in {(total_time - train_start_time) / 60:.2f} minutes.")

    tb_logger.close()
    if tb_logger.enabled:
        logger.info("TensorBoard logger closed.")
