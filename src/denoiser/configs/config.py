from dataclasses import dataclass, field

import torch


@dataclass
class PairingKeyWords:
    """Configuration for pairing clean and noisy images.

    Args:
        clean: Keyword to identify clean images in filename.
        noisy: Keyword to identify noisy images in filename (optional).
        detector: List of detector keywords to filter images (optional).
    """

    clean: str
    noisy: str | None = None
    detector: list[str] | None = None


@dataclass
class TrainConfig:
    # Data configuration
    batch_size: int = 8
    cropsize: int = 256
    noise_sigma: float = 0.1

    # Training configuration
    learning_rate: float = 1e-4
    iteration: int = 10000
    interval: int = 1000

    # Model configuration
    pretrain_model_path: str | None = None

    # Directory configuration
    output_dir: str = "./results"
    log_dir: str = "logs"

    # Data pairing configuration
    pairing_keywords: PairingKeyWords | None = None

    # Device (will be set during runtime)
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    tensorboard: bool = False
