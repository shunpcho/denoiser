"""Tests for configuration classes."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from denoiser.configs.config import PairingKeyWords, TrainConfig


@pytest.mark.parametrize(
    ("clean", "noisy", "detector"),
    [
        ("clean", None, None),
        ("clean", "noisy", None),
        ("clean", None, ["iso3200", "iso1600"]),
        ("clean", "noisy", ["iso3200", "iso1600"]),
        ("clean2", "noisy2", ["iso3200"]),
    ],
)
def test_paring_keywords(clean: str, noisy: str, detector: list[str]) -> None:
    """Test PairingKeyWords configuration."""
    keywords = PairingKeyWords(clean=clean, noisy=noisy, detector=detector)
    assert keywords.clean == clean
    assert keywords.noisy == noisy
    assert keywords.detector == detector


class TestTrainConfig:
    """Test TrainConfig configuration class."""

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({}, TypeError),
            # One of them is missing.
            ({"batch_size": 16, "crop_size": 128, "learning_rate": 1e-3, "iteration": 1000}, TypeError),
            ({"batch_size": 16, "crop_size": 128, "learning_rate": 1e-3, "interval": 100}, TypeError),
            ({"batch_size": 16, "crop_size": 128, "iteration": 1000, "interval": 100}, TypeError),
            ({"batch_size": 16, "learning_rate": 1e-3, "iteration": 1000, "interval": 100}, TypeError),
            ({"crop_size": 128, "learning_rate": 1e-3, "iteration": 1000, "interval": 100}, TypeError),
        ],
    )
    @staticmethod
    def test_train_config_missing_required_fields(
        kwargs: dict,
        expected: type,
    ) -> None:
        """Test that TrainConfig raises error when required fields are missing."""
        with pytest.raises(expected):
            TrainConfig(**kwargs)

    @pytest.mark.parametrize(
        ("batch_size", "crop_size", "model_name", "learning_rate", "iteration", "interval", "kwargs", "expected"),
        [
            # Test with only required fields (default values)
            (
                16,
                128,
                "unet",
                1e-3,
                1000,
                100,
                {},
                {"noise_sigma": 0.1, "output_dir": Path("./results"), "log_dir": Path("logs")},
            ),
            # Test with custom default values
            (
                8,
                64,
                "dncnn",
                5e-4,
                2000,
                200,
                {"noise_sigma": 0.05, "output_dir": Path("./custom_results"), "log_dir": Path("custom_logs")},
                {"noise_sigma": 0.05, "output_dir": Path("./custom_results"), "log_dir": Path("custom_logs")},
            ),
        ],
    )
    @staticmethod
    def test_train_config_creation(
        batch_size: int,
        crop_size: int,
        model_name: str,
        learning_rate: float,
        iteration: int,
        interval: int,
        kwargs: dict,
        expected: dict,
    ) -> None:
        """Test TrainConfig creation with various parameter combinations."""
        config = TrainConfig(
            batch_size=batch_size,
            crop_size=crop_size,
            model_name=model_name,
            learning_rate=learning_rate,
            iteration=iteration,
            interval=interval,
            **kwargs,
        )

        assert config.batch_size == batch_size
        assert config.crop_size == crop_size
        assert config.learning_rate == learning_rate
        assert config.iteration == iteration
        assert config.interval == interval
        assert config.noise_sigma == expected["noise_sigma"]
        assert config.output_dir == expected["output_dir"]
        assert config.log_dir == expected["log_dir"]

    @pytest.mark.parametrize(
        ("attribute", "new_value", "expected_exception"),
        [
            ("batch_size", 8, AttributeError),
            ("crop_size", 64, AttributeError),
            ("learning_rate", 1e-3, AttributeError),
        ],
    )
    @staticmethod
    def test_train_config_frozen_immutable(attribute: str, new_value: float, expected_exception: type) -> None:
        """Test that TrainConfig is immutable (frozen)."""
        config = TrainConfig(
            batch_size=4,
            crop_size=32,
            model_name="unet",
            learning_rate=1e-4,
            iteration=500,
            interval=50,
        )

        # Should not be able to modify frozen dataclass
        with pytest.raises(expected_exception):
            setattr(config, attribute, new_value)

    @pytest.mark.parametrize(
        ("attribute_name", "attribute_value", "expected_exception"),
        [
            ("new_attribute", "test", (AttributeError, TypeError)),
            ("another_attr", 123, (AttributeError, TypeError)),
            ("custom_field", True, (AttributeError, TypeError)),
        ],
    )
    @staticmethod
    def test_train_config_slots(
        attribute_name: str, attribute_value: str | int | bool, expected_exception: type
    ) -> None:
        """Test that TrainConfig uses slots for memory efficiency."""
        config = TrainConfig(
            batch_size=4,
            crop_size=32,
            model_name="dncnn",
            learning_rate=1e-4,
            iteration=500,
            interval=50,
        )

        # Should not be able to add new attributes due to slots
        with pytest.raises(expected_exception):
            setattr(config, attribute_name, attribute_value)

    @pytest.mark.parametrize(("batch_size", "crop_size"), [(2, 64), (16, 256), (32, 128)])
    @staticmethod
    def test_train_config_different_parameters(batch_size: int, crop_size: int) -> None:
        """Test TrainConfig with different parameter combinations."""
        config = TrainConfig(
            batch_size=batch_size,
            crop_size=crop_size,
            model_name="nafnet",
            learning_rate=1e-4,
            iteration=1000,
            interval=100,
        )

        assert config.batch_size == batch_size
        assert config.crop_size == crop_size

    @pytest.mark.parametrize(
        ("output_dir", "log_dir", "expected_output", "expected_log"),
        [
            # Test with Path objects
            (Path("/tmp/output"), Path("/tmp/logs"), Path("/tmp/output"), Path("/tmp/logs")),
            # Test with string paths converted to Path objects
            (Path("./string_results"), Path("string_logs"), Path("./string_results"), Path("string_logs")),
            # Test with mixed types
            (Path("/custom/output"), Path("mixed_logs"), Path("/custom/output"), Path("mixed_logs")),
        ],
    )
    @staticmethod
    def test_train_config_path_handling(
        output_dir: Path, log_dir: Path, expected_output: Path, expected_log: Path
    ) -> None:
        """Test TrainConfig handles Path objects and string conversions correctly."""
        config = TrainConfig(
            batch_size=4,
            crop_size=32,
            model_name="resnet",
            learning_rate=1e-4,
            iteration=500,
            interval=50,
            output_dir=output_dir,
            log_dir=log_dir,
        )

        assert isinstance(config.output_dir, Path)
        assert isinstance(config.log_dir, Path)
        assert config.output_dir == expected_output
        assert config.log_dir == expected_log

    @pytest.mark.parametrize(
        ("batch_size", "crop_size", "learning_rate", "iteration", "interval"),
        [
            (4, 32, 1e-4, 500, 50),
            (8, 64, 1e-3, 1000, 100),
            (16, 128, 5e-4, 2000, 200),
        ],
    )
    @staticmethod
    def test_train_config_reasonable_defaults(
        batch_size: int, crop_size: int, learning_rate: float, iteration: int, interval: int
    ) -> None:
        """Test that default values are reasonable."""
        config = TrainConfig(
            batch_size=batch_size,
            crop_size=crop_size,
            model_name="unet",
            learning_rate=learning_rate,
            iteration=iteration,
            interval=interval,
        )

        # Noise sigma should be reasonable for image denoising
        assert 0.01 <= config.noise_sigma <= 1.0

        # Paths should be reasonable defaults
        assert "results" in str(config.output_dir)
        assert "logs" in str(config.log_dir)

    @pytest.mark.parametrize("noise_sigma", [0.05, 0.1, 0.2, 0.5])
    @staticmethod
    def test_train_config_noise_sigma_range(noise_sigma: float) -> None:
        """Test TrainConfig with different noise sigma values."""
        config = TrainConfig(
            batch_size=4,
            crop_size=32,
            model_name="dncnn",
            learning_rate=1e-4,
            iteration=500,
            interval=50,
            noise_sigma=noise_sigma,
        )

        assert config.noise_sigma == noise_sigma
        assert config.noise_sigma > 0


class TestConfigIntegration:
    """Integration tests for configuration classes."""

    @pytest.mark.parametrize(
        ("clean", "noisy", "detector", "expected_clean", "expected_noisy", "expected_in_detector"),
        [
            ("real", "mean", ["iso3200"], "real", "mean", "iso3200"),
            ("clean", "noisy", ["iso1600", "iso3200"], "clean", "noisy", "iso1600"),
            ("ground_truth", "synthetic", ["iso800"], "ground_truth", "synthetic", "iso800"),
        ],
    )
    @staticmethod
    def test_pairing_keywords_in_train_config_context(
        clean: str,
        noisy: str,
        detector: list[str],
        expected_clean: str,
        expected_noisy: str,
        expected_in_detector: str,
    ) -> None:
        """Test using PairingKeyWords in training context."""
        pairing = PairingKeyWords(clean=clean, noisy=noisy, detector=detector)

        # Would typically be used in training configuration
        assert pairing.clean == expected_clean
        assert pairing.noisy == expected_noisy
        assert expected_in_detector in pairing.detector  # type: ignore[operator]

    @pytest.mark.parametrize(
        ("batch_size", "crop_size", "learning_rate", "iteration", "interval", "expected_batch", "expected_noise"),
        [
            (8, 64, 1e-3, 1000, 100, 8, 0.1),
            (16, 128, 5e-4, 2000, 200, 16, 0.1),
            (4, 32, 1e-4, 500, 50, 4, 0.1),
        ],
    )
    @staticmethod
    def test_config_serialization_compatibility(
        batch_size: int,
        crop_size: int,
        learning_rate: float,
        iteration: int,
        interval: int,
        expected_batch: int,
        expected_noise: float,
    ) -> None:
        """Test that configs work with common serialization."""
        config = TrainConfig(
            batch_size=batch_size,
            crop_size=crop_size,
            model_name="unet",
            learning_rate=learning_rate,
            iteration=iteration,
            interval=interval,
        )

        # Should be able to convert to dict (for serialization)
        config_dict = {
            "batch_size": config.batch_size,
            "crop_size": config.crop_size,
            "learning_rate": config.learning_rate,
            "iteration": config.iteration,
            "interval": config.interval,
            "noise_sigma": config.noise_sigma,
        }

        assert config_dict["batch_size"] == expected_batch
        assert config_dict["noise_sigma"] == expected_noise

    @pytest.mark.parametrize(
        ("batch_size", "crop_size", "learning_rate", "iteration", "interval"),
        [
            (4, 32, 1e-4, 500, 50),
            (8, 64, 1e-3, 1000, 100),
            (16, 128, 5e-4, 2000, 200),
        ],
    )
    @staticmethod
    def test_config_types_compatibility(
        batch_size: int, crop_size: int, learning_rate: float, iteration: int, interval: int
    ) -> None:
        """Test config compatibility with PyTorch types."""
        config = TrainConfig(
            batch_size=batch_size,
            crop_size=crop_size,
            model_name="nafnet",
            learning_rate=learning_rate,
            iteration=iteration,
            interval=interval,
        )

        # Should be compatible with PyTorch operations
        learning_rate_tensor = torch.tensor(config.learning_rate)
        assert isinstance(learning_rate_tensor, torch.Tensor)
        assert learning_rate_tensor.item() == pytest.approx(config.learning_rate)
