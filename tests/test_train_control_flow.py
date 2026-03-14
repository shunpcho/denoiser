import logging
import sys
from pathlib import Path

import torch

# Ensure package source is importable when tests run from workspace root
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "denoiser" / "src"))

import denoiser.train as train_module
from denoiser.configs.config import TrainConfig


def test_train_control_flow_monkeypatched(tmp_path, monkeypatch):
    # Patch create_model to return a tiny model with parameters
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return x

    monkeypatch.setattr(
        train_module, "create_model", lambda model_name, in_channels, out_channels, pretrained: DummyModel()
    )
    monkeypatch.setattr(train_module, "load_model_checkpoint", lambda model, path, device: model)

    # Dummy DataLoader to avoid worker processes
    class DummyDataLoader:
        def __init__(self, dataset, batch_size, shuffle, num_workers, pin_memory, persistent_workers, collate_fn):
            self.sample = (torch.randn(1, 3, 8, 8), torch.randn(1, 3, 8, 8))

        def __iter__(self):
            yield self.sample

        def __len__(self):
            return 1

    monkeypatch.setattr(train_module.torch.utils.data, "DataLoader", DummyDataLoader)
    # Patch datasets to avoid filesystem access

    class DummyDataset:
        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 1

    monkeypatch.setattr(train_module, "PairedDataset", DummyDataset)
    monkeypatch.setattr(train_module, "TiledPairedDataset", DummyDataset)

    # Fake trainer that records save_model calls and returns controlled losses
    saved = {}

    class FakeTrainer:
        def __init__(self, models, optimizer, train_config, train_dataloader, val_dataloader):
            self.models = models
            self.optimizer = optimizer
            self.train_config = train_config
            self.train_dataloader = train_dataloader
            self.val_dataloader = val_dataloader
            self.val_calls = 0

        def train_step(self):
            return {"Loss": 0.2, "MSE": 0.2}

        def val_step(self):
            self.val_calls += 1
            if self.val_calls == 1:
                return {"Loss": 0.1, "PSNR": 30.0}
            return {"Loss": 0.2}

        def save_model(self, path):
            saved["path"] = str(path)

    monkeypatch.setattr(train_module, "TrainTrainer", FakeTrainer)

    # Patch TensorBoard and other IO heavy utilities
    class FakeTB:
        def __init__(self, log_dir, dataloader, device, crop_size, destandardize_img_fn, max_outputs):
            self.logged_graph = False

        def log_model_graph(self, model, sample_input):
            self.logged_graph = True

        def log_training_metrics(self, **kwargs):
            pass

        def log_images(self, *args, **kwargs):
            pass

        def log_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(train_module, "TensorBoard", FakeTB)
    monkeypatch.setattr(train_module, "save_validation_predictions_stitched", lambda **kwargs: None)
    monkeypatch.setattr(train_module, "create_logger", lambda name, log_dir, verbose: logging.getLogger("test"))

    # Build TrainConfig
    config = TrainConfig.from_optional_kwargs(
        batch_size=1,
        crop_size=8,
        model_name="dummy",
        noise_sigma=0.0,
        learning_rate=1e-4,
        loss_type="mse",
        iteration=2,
        interval=1,
        pretrain_model_path=None,
        output_dir=tmp_path,
        log_dir="logs",
        pairing_keywords=None,
        tensorboard=True,
    )

    # Run train (should use all patched objects and not spawn workers)
    train_module.train(train_data_path=tmp_path, train_config=config, val_data_path=tmp_path, limit=1, verbose="info")

    # Assert that model was saved at least once
    assert "path" in saved
    assert Path(saved["path"]).name == "best_model.pth"
