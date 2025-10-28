import argparse
import json
import random
from pathlib import Path

from denoiser.configs.config import PairingKeyWords


def split_train_val(
    data_dir: Path,
    train_ratio: float = 0.8,
    img_read_keywords: PairingKeyWords | None = None,
) -> None:
    """Split dataset into training and validation sets and save the file lists as JSON files.

    Args:
        data_dir: Directory containing the images.
        train_ratio: Ratio of training data to total data.
        img_read_keywords: Keywords to filter images based on filenames.
    """
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif"]

    files = [p for p in data_dir.rglob("*") if p.suffix.lower() in exts and _match_keywords(p, img_read_keywords)]

    # Split into train and val
    random.seed(42)
    random.shuffle(files)
    split_idx = int(len(files) * train_ratio)
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    # save to JSON
    out_dir = data_dir / "indices"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "train_list.json").open("w", encoding="utf-8") as f:
        json.dump([str(p) for p in train_files], f, indent=2, ensure_ascii=False)
    with (out_dir / "val_list.json").open("w", encoding="utf-8") as f:
        json.dump([str(p) for p in val_files], f, indent=2, ensure_ascii=False)

    print(f"train: {len(train_files)}, val: {len(val_files)}")


def _match_keywords(p: Path, keywords: PairingKeyWords | None) -> bool:
    if keywords is None:
        return True

    if keywords.detector is not None:
        return keywords.clean in p.stem and keywords.detector[0] in p.stem
    return keywords.clean in p.stem


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True, help="Path to the training data.")
    parser.add_argument(
        "--clean_img_keyword", type=str, default=None, help="Keyword to identify clean images in filename."
    )
    parser.add_argument(
        "--noisy_img_keyword", type=str, default=None, help="Keyword to identify noisy images in filename."
    )
    parser.add_argument(
        "--detector_keywords", type=str, nargs="*", default=None, help="List of detector keywords (optional)."
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
    split_train_val(**args, img_read_keywords=pairing_keywords)
