# Denoiser

A framework for deep learning based image restoration, such as noise removal and blur removal.

## Overview

- Research optimal solutions across various models and architectures (currently supports U-Net architecture)
- Handles both color (RGB) and grayscale images

## Development

### Requirements

If you use uv, you can install the dependencies easily:

- uv >= 0.8.3

## Usage

### Dataset Requirements

The framework supports both color (RGB) and grayscale images.

**Dataset Preparation:**

- Prepare dataset with (clean, noisy) image pairs for training
- Alternatively, use only clean images and add synthetic Gaussian noise
- Image pairs must have the same base name with different keywords (e.g., `xxx_clean.png` and `xxx_noisy.png`)

<details>
<summary>Sample Dataset Preparation</summary>

```bash
sh script/sample_dataset.sh download unzip
```

- First flag: download the dataset if needed
- Second flag: unzip the downloaded dataset if needed

**Dataset Structure:**

- `_ours`: output image
- `_mean`: clean image
- `_real`: noisy image
- Contains 15 image sets

```
  data/CC15
  |-- xxx_ours.png
  |-- xxx_mean.png
  |-- xxx_real.png
  |-- yyy_ours.png
  |-- yyy_mean.png
  |-- yyy_real.png
  :
  :
  `-- zzz_real.png
```

</details>

### Split dataset into train and val and Make list of file paths to json file

- Run script and save the list

```bash
python src/denoiser/make_train_val_split.py \
--data_dir data/CC15 \
--clean_img_keyword _mean \
--noisy_img_keyword _real
```

- This generates train/validation split files (80/20 ratio by default):

```
data/<dataset name>
|-- indices
|   |-- train_list.json
|   `-- val_list.json
|-- ~_mean.png
|-- ~_real.png
:
```

The script uses a fixed random seed (42) for reproducible splits.


### Training

Example command for training with sample data:

```bash
python src/denoiser/train.py \
    --train_data_path "data/CC15" \
    --clean_img_keyword "_mean" \
    --noisy_img_keyword "_real" \
    --batch_size 4 \
    --cropsize 256 \
    --learning_rate 0.0001 \
    --iteration 300 \
    --interval 50 \
    --output_dir "./results" \
    --log_dir "logs" \
    --tensorboard True \
    --verbose "info"

```

<details>
<summary>Training options</summary>

| Args                    | Type  | Default     | Detail                                       |
| ----------------------- | ----- | ----------- | -------------------------------------------- |
| `--train_data_path`     | Path  | Required    | Path to the training data directory          |
| `--valid_data_path`     | Path  | None        | Path to the validation data (optional)       |
| `--clean_img_keyword`   | str   | None        | Keyword to identify clean images in filename |
| `--noisy_img_keyword`   | str   | None        | Keyword to identify noisy images in filename |
| `--output_dir`          | Path  | "./results" | Directory to save results                    |
| `--log_dir`             | Path  | "logs"      | Directory to save logs                       |
| `--batch_size`          | int   | 4           | Training batch size                          |
| `--cropsize`            | int   | None        | Crop size for training images                |
| `--noise_sigma`         | float | None        | Standard deviation of Gaussian noise         |
| `--learning_rate`       | float | 1e-4        | Learning rate for optimizer                  |
| `--iteration`           | int   | 1000        | Number of training iterations                |
| `--interval`            | int   | 100         | Validation interval                          |
| `--pretrain_model_path` | Path  | None        | Path to the pre-trained model                |
| `--tensorboard`         | bool  | True        | Enable TensorBoard logging                   |
| `--verbose`             | str   | "info"      | Logging verbosity level (debug/info/error)   |

</details>
