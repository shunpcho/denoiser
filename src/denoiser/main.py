#!/usr/bin/env python3
"""Main script to run denoiser training."""

from pathlib import Path

from denoiser.configs.config import PairingKeyWords, TrainConfig
from denoiser.train import train


def main() -> None:
    """Run the training process with default configuration."""
    # Create pairing keywords configuration
    pairing_keywords = PairingKeyWords(
        clean="mean",
        noisy="real",
        detector=None,
    )

    # Create training configuration
    config = TrainConfig(
        batch_size=4,
        cropsize=128,  # Smaller size for faster training during development
        noise_sigma=0.1,
        learning_rate=1e-4,
        iteration=50,  # Reduced for testing
        interval=5,  # More frequent validation
        output_dir="./results",
        log_dir="logs",
        pairing_keywords=pairing_keywords,
        tensorboard=True,
    )

    # Set up data paths (adjust these paths according to your data structure)
    train_data_path = Path("/home/s.chochi/noise-translator/data/CC15")

    print("Starting denoiser training...")
    print(f"Configuration: {config}")
    print(f"Train data path: {train_data_path}")

    # Run training
    try:
        train(
            train_data_path=train_data_path,
            train_config=config,
            limit=None,  # No limit on samples
            verbose="info",
        )
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
