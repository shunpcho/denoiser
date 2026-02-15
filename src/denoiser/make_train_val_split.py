import argparse
import json
import random
from pathlib import Path
from typing import TypedDict

from denoiser.configs.config import PairingKeyWords


class TrainValSplitKwargs(TypedDict):
    train_path: Path
    val_path: Path


def split_train_val(
    data_dir: Path,
    train_val: TrainValSplitKwargs | float | None = None,
    img_read_keywords: PairingKeyWords | None = None,
) -> None:
    """Split dataset into training and validation sets and save the file lists as JSON files.

    Args:
        data_dir: Directory containing the images.
        img_read_keywords: Keywords to filter images based on filenames.
        train_val: Optional paths for train and validation splits or a float for train ratio.

    Raises:
        TypeError: If train_val is not a TrainValSplitKwargs dict or a floats for train ratio.
    """
    if isinstance(train_val, dict):
        train_data_dir = data_dir / train_val["train_path"]
        val_data_dir = data_dir / train_val["val_path"]
        train_files = _filter_files_by_keywords(train_data_dir, img_read_keywords)
        val_files = _filter_files_by_keywords(val_data_dir, img_read_keywords)
    elif isinstance(train_val, float):
        files = _filter_files_by_keywords(data_dir, img_read_keywords)
        # Split into train and val
        random.seed(42)
        random.shuffle(files)
        split_idx = int(len(files) * train_val)
        train_files = files[:split_idx]
        val_files = files[split_idx:]
    else:
        msg = "train_val must be either a TrainValSplitKwargs dict or a float for train ratio."
        raise TypeError(msg)

    # save to JSON
    out_dir = data_dir / "indices"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "train_list.json").open("w", encoding="utf-8") as f:
        json.dump([str(p) for p in train_files], f, indent=2, ensure_ascii=False)
    with (out_dir / "val_list.json").open("w", encoding="utf-8") as f:
        json.dump([str(p) for p in val_files], f, indent=2, ensure_ascii=False)

    print(f"train: {len(train_files)}, val: {len(val_files)}")


def _filter_files_by_keywords(data_dir: Path, img_read_keywords: PairingKeyWords | None) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
    return [p for p in data_dir.rglob("*") if p.suffix.lower() in exts and _match_keywords(p, img_read_keywords)]


def _match_keywords(p: Path, keywords: PairingKeyWords | None) -> bool:
    if keywords is None:
        return True

    if keywords.detector is not None:
        return keywords.clean in p.stem and keywords.detector[0] in p.stem
    return keywords.clean in p.stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to the training data.")
    parser.add_argument(
        "--clean-img-keyword", type=str, default=None, help="Keyword to identify clean images in filename."
    )
    parser.add_argument(
        "--noisy-img-keyword", type=str, default=None, help="Keyword to identify noisy images in filename."
    )
    parser.add_argument(
        "--detector-keywords", type=str, nargs="*", default=None, help="List of detector keywords (optional)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--train-val-path", type=Path, nargs=2, help="Paths to train and val if using directory-based split."
    )
    group.add_argument("--train-ratio", type=float, help="Ratio of training data (e.g., 0.8 for 80% train, 20% val).")

    args = parser.parse_args()
    args = vars(args)

    clean_img_keyword = args.pop("clean_img_keyword")
    noisy_img_keyword = args.pop("noisy_img_keyword")
    detector_keywords = args.pop("detector_keywords")

    pairing_keywords = (
        PairingKeyWords(
            clean=clean_img_keyword,
            noisy=noisy_img_keyword,
            detector=detector_keywords or None,
        )
        if clean_img_keyword
        else None
    )
    train_val_path = args.pop("train_val_path")
    train_ratio = args.pop("train_ratio")

    if train_val_path:
        train_val = TrainValSplitKwargs(
            train_path=train_val_path[0],
            val_path=train_val_path[1],
        )
    elif train_ratio is not None:
        train_val = train_ratio
    else:
        msg = "Either --train-val-path or --train-ratio must be provided."
        raise ValueError(msg)
    print(args)
    split_train_val(**args, train_val=train_val, img_read_keywords=pairing_keywords)


if __name__ == "__main__":
    main()
