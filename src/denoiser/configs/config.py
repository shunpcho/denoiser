from dataclasses import dataclass, field
from pathlib import Path
from typing import Self, TypedDict, Unpack

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


class _TrainConfigKwargs(TypedDict, total=False):
    batch_size: int
    cropsize: int
    noise_sigma: float
    learning_rate: float
    iteration: int
    interval: int
    pretrain_model_path: str | None
    output_dir: str
    log_dir: str
    pairing_keywords: PairingKeyWords | None
    tensorboard: bool
    device: torch.device


@dataclass(frozen=True, slots=True)
class TrainConfig:
    # Required fields (no defaults) must come first
    batch_size: int
    cropsize: int
    learning_rate: float
    iteration: int
    interval: int

    # Optional fields with defaults
    noise_sigma: float = 0.1
    output_dir: Path = Path("./results")
    log_dir: Path = Path("logs")

    pairing_keywords: PairingKeyWords | None = None

    pretrain_model_path: str | None = None

    tensorboard: bool = False

    # Device (will be set during runtime)
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    @classmethod
    def from_optional_kwargs(cls, **kwargs: Unpack[_TrainConfigKwargs]) -> Self:
        """Create a TrainConfig instance from given keyword arguments.

        This method allows creating a TrainConfig instance by providing only
        the desired parameters, while the rest will take default values.

        Args:
            **kwargs: Keyword arguments corresponding to TrainConfig fields.

        Returns:
            An instance of TrainConfig with specified and default values.
        """
        return cls(**{key: value for key, value in kwargs.items() if value is not None})
