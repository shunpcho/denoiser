from dataclasses import dataclass, field
from pathlib import Path
from typing import Self, TypedDict, Unpack

import torch

TB_VALID_ITEMS = frozenset({"metrics", "images", "graph", "histograms", "weights-analysis"})
TB_VALID_METRIC_TAGS = frozenset(
    {"train-mse", "val-mse", "train-psnr", "val-psnr", "train-ssim", "val-ssim", "train-esfl", "val-esfl"}
)


class _TensorboardConfigKwargs(TypedDict, total=False):
    enabled: bool
    max_outputs: int
    log_subdir: Path
    items: frozenset[str]
    metric_tags: frozenset[str]


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


@dataclass(frozen=True, slots=True)
class TensorboardConfig:
    """Configuration for TensorBoard logging.

    Args:
        enabled: Whether to enable TensorBoard logging.
        max_outputs: Maximum number of sample images to log.
        log_subdir: Subdirectory under output_dir for TensorBoard logs.
        items: Output categories to enable. Valid: metrics, images, graph,
            histograms, weights-analysis. Defaults to {metrics, images}.
        metric_tags: Per-tag scalars beyond train/val loss. Valid:
            train-mse, val-mse, train-psnr, val-psnr, train-ssim,
            val-ssim, train-esfl, val-esfl. Defaults to empty.
    """

    enabled: bool = True
    max_outputs: int = 4
    log_subdir: Path = Path("tensorboard")
    items: frozenset[str] = field(default_factory=lambda: frozenset({"metrics", "images"}))
    metric_tags: frozenset[str] = field(default_factory=frozenset)  # pyright: ignore[reportUnknownVariableType]

    @classmethod
    def from_optional_kwargs(cls, **kwargs: Unpack[_TensorboardConfigKwargs]) -> Self:
        """Create a TensorboardConfig instance from optional keyword arguments."""
        return cls(**{key: value for key, value in kwargs.items() if value is not None})  # pyright: ignore[reportArgumentType]


class _TrainConfigKwargs(TypedDict, total=False):
    batch_size: int
    crop_size: int
    model_name: str
    noise_sigma: float
    learning_rate: float
    loss_type: str
    iteration: int
    interval: int
    pretrain_model_path: Path | None
    output_dir: Path
    log_dir: Path
    pairing_keywords: PairingKeyWords | None
    tensorboard_config: TensorboardConfig
    device: torch.device


@dataclass(frozen=True, slots=True)
class TrainConfig:
    # Required fields (no defaults) must come first
    batch_size: int
    crop_size: int
    learning_rate: float
    loss_type: str
    iteration: int
    interval: int

    # Optional fields with defaults
    noise_sigma: float = 0.1
    output_dir: Path = Path("./results")
    log_dir: Path = Path("logs")

    model_name: str | None = None

    pairing_keywords: PairingKeyWords | None = None

    pretrain_model_path: Path | None = None

    tensorboard_config: TensorboardConfig = field(default_factory=TensorboardConfig)

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
        return cls(**{key: value for key, value in kwargs.items() if value is not None})  # pyright: ignore[reportArgumentType]
