from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.utils.data
from torch import optim

from denoiser.configs.config import PairingKeyWords, TrainConfig
from denoiser.data.data_loader import PairedDataset, TrainSubset, ValSubset
from denoiser.data.data_transformations import (
    compose_transformations,
    destandardize_img,
    load_img_clean,
    pairing_clean_noisy,
    random_crop,
    standardize_img,
)
from denoiser.models.build_model import create_model, load_model_checkpoint
from denoiser.utils.data_utils import collate_fn
from denoiser.utils.get_logger import create_logger
from denoiser.utils.tensorboard_log import TensorBoard
from denoiser.utils.trainer import TrainTrainer
from denoiser.utils.vis_img import save_validation_predictions


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

    logger.info(f"Training data path: {train_data_path}")
    if valid_data_path:
        logger.info(f"Validation data path: {valid_data_path}")
    logger.info(f"Training configuration: {train_config}")

    logger.info("Created output and log directories.")

    device = train_config.device

    img_channels = (
        3
        if train_config.pairing_keywords is None or train_config.pairing_keywords.detector is None
        else len(train_config.pairing_keywords.detector)
    )

    if pretrain_model_path := train_config.pretrain_model_path:
        if not pretrain_model_path.is_file():
            msg = f"Pretrained model file not found: {pretrain_model_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info(f"Using pretrained model from: {pretrain_model_path}")

    logger.info("Data loading and preprocessing...")

    pairing_words = train_config.pairing_keywords  # Optional[PairingKeywords]
    noise_sigma = train_config.noise_sigma
    # Set default crop size if not specified
    crop_size = train_config.cropsize if train_config.cropsize is not None else 256

    clean_img_loader = load_img_clean(pairing_words)
    noisy_img_loader = pairing_clean_noisy(pairing_words, noise_sigma)

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
    mean = (0.0,) * img_channels
    std = (1.0,) * img_channels
    standardization_fn = standardize_img(mean, std)
    destandardize_img_fn = destandardize_img(mean, std)

    dataset = PairedDataset(
        train_data_path,
        data_loading_fn=clean_img_loader,
        img_standardization_fn=standardization_fn,
        pairing_fn=noisy_img_loader,
        data_augmentation_fn=augmentation_fn,
        img_read_keywords=pairing_words,
        noise_sigma=noise_sigma,
        limit=limit,
    )
    dataset_size = len(dataset)
    logger.info(f"Dataset size: {dataset_size} samples")
    train_size = int(0.8 * dataset_size)
    valid_size = dataset_size - train_size
    train_subset, valid_subset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    logger.info(f"Train/Validation split: {train_size}/{valid_size} samples")

    train_dataset = TrainSubset(dataset, train_subset.indices)
    val_dataset = ValSubset(dataset, valid_subset.indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    logger.info("Data loaders created.")

    # Build models...
    # models = UNet(in_ch=img_channels, out_ch=img_channels, base_ch=64).to(device)
    models = create_model(
        model_name=train_config.model_name, in_channels=img_channels, out_channels=img_channels, pretrained=True
    ).to(device)

    if not pretrain_model_path:
        logger.info("No pretrained model specified, initializing new model.")
    else:
        logger.info(f"Loading pretrained model from {pretrain_model_path}")
        models = load_model_checkpoint(models, pretrain_model_path, device)

    optimizer = optim.Adam(models.parameters(), lr=train_config.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Debug: Check data types and shapes
    sample_batch = next(iter(val_loader))
    clean_sample, _ = sample_batch
    input_shape = clean_sample.shape
    logger.info("Start training ...")
    logger.info(f"Input shape: {input_shape}")

    trainer = TrainTrainer(
        models,
        optimizer,
        train_config,
        train_loader,
        val_loader,
    )

    tblog_dir = output_dir / "tensorboard"
    tb_logger = TensorBoard(
        log_dir=tblog_dir,
        dataloader=val_loader,
        device=device,
        crop_size=train_config.cropsize,
        destandardize_img_fn=destandardize_img_fn,
        max_outputs=4,
    )
    if train_config.tensorboard:
        logger.info(f"TensorBoard logging enabled: {tblog_dir}")

    if train_config.tensorboard:
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

            tb_logger.log_images(models, step=iteration, tag_prefix="Sample")

            # Save validation prediction images
            pred_images_dir = output_dir / "validation_predictions"
            save_validation_predictions(
                model=models,
                val_loader=val_loader,
                device=device,
                destandardize_fn=destandardize_img_fn,
                save_dir=pred_images_dir,
                iteration=iteration,
                max_samples=4,
            )
            logger.info(f"Validation prediction images saved to {pred_images_dir}")

            models.train()
            scheduler.step()

    total_time = time.time()
    logger.info(f"Training completed in {(total_time - train_start_time) / 60:.2f} minutes.")

    tb_logger.close()
    if train_config.tensorboard:
        logger.info("TensorBoard logger closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a denoiser model.")
    parser.add_argument("--train_data_path", type=Path, required=True, help="Path to the training data.")
    parser.add_argument("--valid_data_path", type=Path, default=None, help="Path to the validation data (optional).")

    parser.add_argument(
        "--clean_img_keyword", type=str, default=None, help="Keyword to identify clean images in filename."
    )
    parser.add_argument(
        "--noisy_img_keyword", type=str, default=None, help="Keyword to identify noisy images in filename."
    )
    parser.add_argument(
        "--detector_keywords", type=str, nargs="*", default=None, help="List of detector keywords (optional)."
    )
    parser.add_argument("--output_dir", type=Path, default="./results", help="Directory to save results.")
    parser.add_argument("--log_dir", type=Path, default="logs", help="Directory to save logs.")
    parser.add_argument("--model_name", type=str, default="resnet34", help="Model architecture to use.")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--cropsize", type=int, default=None, help="Crop size for training images.")
    parser.add_argument("--noise_sigma", type=float, default=None, help="Standard deviation of Gaussian noise.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer.")
    parser.add_argument("--iteration", type=int, default=1000, help="Number of training iterations.")
    parser.add_argument("--interval", type=int, default=100, help="Validation interval.")
    parser.add_argument("--limit", type=int, default=None, help="Limit on number of training samples (optional).")
    parser.add_argument("--pretrain_model_path", type=Path, default=None, help="Path to the pre-trained model.")
    parser.add_argument("--tensorboard", action="store_true", default=True, help="Enable TensorBoard logging.")
    parser.add_argument(
        "--verbose", type=str, choices=["debug", "info", "error"], default="info", help="Logging verbosity level."
    )

    args = parser.parse_args()
    args = vars(args)

    pairing_keywords = (
        PairingKeyWords(
            clean=args.pop("clean_img_keyword"),
            noisy=args.pop("noisy_img_keyword"),
            detector=args.pop("detector_keywords") or None,
        )
        if args.get("clean_img_keyword")
        else None
    )

    """Run the training process with default configuration."""

    # Create training configuration
    config = TrainConfig.from_optional_kwargs(
        batch_size=args.pop("batch_size"),
        cropsize=args.pop("cropsize"),
        model_name=args.pop("model_name"),
        noise_sigma=args.pop("noise_sigma"),
        learning_rate=args.pop("learning_rate"),
        iteration=args.pop("iteration"),
        interval=args.pop("interval"),
        pretrain_model_path=args.pop("pretrain_model_path"),
        output_dir=args.pop("output_dir"),
        log_dir=args.pop("log_dir"),
        pairing_keywords=pairing_keywords,
        tensorboard=args.pop("tensorboard"),
    )

    # Run training
    try:
        train(**args, train_config=config)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
