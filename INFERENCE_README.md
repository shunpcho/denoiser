# Denoiser Inference

This module provides functionality to run inference using trained denoiser models.

## Files

- `inference.py` - Main inference module with functions for loading models and denoising images
- `example_inference.py` - Example script showing how to use the inference functionality

## Usage

### Command Line Interface

You can use the inference module directly from the command line:

```bash
# Denoise a single image
python -m denoiser.inference --model results/best_model.pth --input noisy_image.jpg --output denoised_image.jpg

# Denoise all images in a directory
python -m denoiser.inference --model results/best_model.pth --input ./noisy_images/ --output ./denoised_images/ --batch

# Use CPU instead of GPU
python -m denoiser.inference --model results/best_model.pth --input noisy_image.jpg --output denoised_image.jpg --device cpu
```

### Python API

You can also use the inference functions directly in your Python code:

```python
import torch
from pathlib import Path
from denoiser.inference import load_model, denoise_image

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(Path("results/best_model.pth"), device)

# Denoise single image
denoise_image(model, Path("noisy.jpg"), Path("denoised.jpg"), device)
```

### Batch Processing

For processing multiple images:

```python
from denoiser.inference import batch_denoise

# Process all images in a directory
batch_denoise(model, Path("./input_dir"), Path("./output_dir"), device)
```

## Model Requirements

The inference module expects model checkpoints saved in the format used by the training script:

```python
{
    "model_state_dict": model.state_dict(),
    # ... other training metadata
}
```

## Image Processing

The inference module handles:
- Loading images in various formats (PNG, JPG, TIFF, etc.)
- Preprocessing (standardization to [0,1] range)
- Model inference
- Postprocessing (destandardization back to [0,255] range)
- Saving results

## Error Handling

The module provides appropriate error handling for:
- Missing model files
- Invalid image files
- CUDA out-of-memory errors
- I/O errors

## Performance Tips

1. Use GPU when available for faster inference
2. Process images in batches when dealing with many files
3. Consider image size - larger images require more memory
4. Use appropriate image formats (PNG for lossless, JPG for smaller files)