# Denoiser Tests

This directory contains comprehensive test suites for the denoiser package.

## Test Structure

```
tests/
├── __init__.py          # Test package initialization
├── conftest.py          # Shared fixtures and pytest configuration
├── test_models.py       # Tests for UNet and model components
├── test_data_transformations.py  # Tests for data transformation functions
├── test_data_loader.py  # Tests for dataset and data loading
├── test_loss.py         # Tests for loss calculation functions
├── test_inference.py    # Tests for inference functionality
└── test_config.py       # Tests for configuration classes
```

## Running Tests

### Prerequisites

Install test dependencies:

```bash
# Using uv (recommended)
uv sync --group test

# Or using pip
pip install -e ".[test]"
```

### Basic Test Execution

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=denoiser --cov-report=html
```

### Using the Test Runner Script

```bash
# Run all tests
python run_tests.py

# Run only fast tests (exclude slow/integration tests)
python run_tests.py --type fast

# Run GPU tests (requires CUDA)
python run_tests.py --type gpu

# Run with coverage report
python run_tests.py --type coverage
```

### Test Categories

Tests are organized with markers for different categories:

- **Fast tests**: Quick unit tests (`pytest -m "not slow"`)
- **Slow tests**: Integration tests and complex operations (`pytest -m slow`)
- **GPU tests**: Tests requiring CUDA (`pytest -m gpu`)
- **Integration tests**: End-to-end functionality tests (`pytest -m integration`)

## Test Coverage

The test suite covers:

### Models (`test_models.py`)

- UNet architecture initialization and forward pass
- ConvBlock functionality
- Model parameter counting and gradient flow
- Device compatibility (CPU/GPU)
- Different input sizes and channel configurations

### Data Processing (`test_data_transformations.py`)

- Image loading functions (RGB, grayscale)
- Image standardization and destandardization
- Data pairing and noise generation
- Transformation composition
- Random cropping and augmentation

### Data Loading (`test_data_loader.py`)

- PairdDataset class functionality
- Gaussian noise function generation
- Dataset initialization with various parameters
- Index handling and path sorting
- Mock function integration

### Loss Functions (`test_loss.py`)

- MSE, PSNR, and SSIM calculations
- Loss function behavior with identical/different images
- Gradient computation and device consistency
- Denoiser loss integration with models

### Inference (`test_inference.py`)

- Model loading and saving
- Image preprocessing and postprocessing
- Single image denoising
- Batch processing functionality
- Device compatibility and error handling

### Configuration (`test_config.py`)

- TrainConfig dataclass validation
- PairingKeyWords configuration
- Immutability and type checking
- Default value validation

## Writing New Tests

### Test Organization

- Group related tests in classes (e.g., `TestUNet`, `TestDataLoader`)
- Use descriptive test names that explain what is being tested
- Follow the pattern: `test_<component>_<behavior>_<condition>`

### Fixtures

- Use `conftest.py` for shared fixtures
- Create component-specific fixtures in test files when needed
- Use `pytest.fixture` for setup/teardown operations

### Assertions

- Use specific assertions (e.g., `pytest.approx()` for floats)
- Test both positive and negative cases
- Verify shapes, types, and value ranges
- Check error conditions with `pytest.raises()`

### Example Test

```python
class TestNewFeature:
    """Test new feature functionality."""

    @pytest.fixture
    def sample_data(self) -> np.ndarray:
        """Create sample test data."""
        return np.random.rand(32, 32, 3)

    def test_feature_basic_functionality(self, sample_data: np.ndarray) -> None:
        """Test basic feature operation."""
        result = new_feature_function(sample_data)

        assert isinstance(result, np.ndarray)
        assert result.shape == sample_data.shape
        assert result.dtype == np.float32

    @pytest.mark.parametrize("param", [1, 2, 4])
    def test_feature_different_parameters(self, param: int) -> None:
        """Test feature with different parameter values."""
        result = new_feature_function(param=param)
        assert result is not None
```

### Markers

Add appropriate markers to tests:

```python
@pytest.mark.slow
def test_expensive_operation():
    """Test that takes significant time."""
    pass

@pytest.mark.gpu
def test_cuda_functionality():
    """Test requiring GPU."""
    pass
```

## Continuous Integration

Tests are designed to work in CI environments:

- CPU-only tests run by default
- GPU tests are skipped if CUDA is not available
- Temporary files are properly cleaned up
- Tests are deterministic with fixed random seeds

## Debugging Tests

### Running Individual Tests

```bash
# Run specific test method
pytest tests/test_models.py::TestUNet::test_unet_forward

# Run with debugging
pytest tests/test_models.py -s --pdb

# Show local variables on failure
pytest tests/ --tb=long
```

### Common Issues

- **Import errors**: Ensure denoiser package is installed in development mode
- **CUDA tests failing**: GPU tests require CUDA-enabled PyTorch
- **File permission errors**: Tests create temporary files - check permissions
- **Random test failures**: Some tests use randomization - check for proper seeding

## Performance Considerations

- Tests use small model sizes and images for speed
- Fixtures create minimal test data
- GPU tests are marked separately to avoid unnecessary device allocation
- Temporary files are cleaned up automatically

## Contributing

When adding new functionality to the denoiser package:

1. Write tests for the new feature
2. Ensure tests pass on both CPU and GPU (if applicable)
3. Add appropriate markers for test categorization
4. Update this README if new test patterns are introduced
