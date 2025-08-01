# Data Parallel Support in ProteinEncoder

This document describes the data parallel functionality added to the `ProteinEncoder` class, which allows for efficient processing across multiple GPUs.

## Overview

Data parallel processing distributes batches across multiple GPUs, allowing for:
- **Increased throughput**: Process larger batches or more sequences simultaneously
- **Better resource utilization**: Make use of all available GPU memory
- **Improved performance**: Reduce total processing time for large datasets

## Requirements

- PyTorch with CUDA support
- Multiple CUDA-capable GPUs
- At least 2 GPUs for meaningful performance improvement

## Usage

### Basic Data Parallel Setup

```python
from protenc.encoder import get_encoder

# Enable data parallel with automatic device selection
encoder = get_encoder(
    model_name="esm2_t6",
    device="cuda:0",
    batch_size=8,  # Larger batch size recommended for data parallel
    data_parallel=True,
    device_ids=None  # Use all available GPUs
)
```

### Specifying Specific GPUs

```python
# Use specific GPU devices
encoder = get_encoder(
    model_name="esmc_300m",
    device="cuda:0",
    batch_size=8,
    data_parallel=True,
    device_ids=[0, 1, 2]  # Use GPUs 0, 1, and 2
)
```

### Direct ProteinEncoder Usage

```python
from protenc.encoder import ProteinEncoder
from protenc.models import get_model

model = get_model("prot_bert")
model = model.to("cuda:0")

encoder = ProteinEncoder(
    model=model,
    batch_size=8,
    data_parallel=True,
    device_ids=[0, 1]
)
```

## Configuration Options

### Constructor Parameters

- `data_parallel` (bool): Enable data parallel processing (default: False)
- `device_ids` (list[int]): List of GPU device IDs to use (default: None, uses all available)

### Batch Size Considerations

When using data parallel:
- **Increase batch size**: Data parallel works best with larger batches
- **Memory distribution**: Batch is split across GPUs, so you can use larger total batch sizes
- **Recommended**: 2-4x the single-GPU batch size

### Device Management

- **Primary device**: The first GPU in `device_ids` (or `cuda:0` if not specified)
- **Input data**: Automatically moved to the primary device
- **Model distribution**: PyTorch's `nn.DataParallel` handles distribution across GPUs

## Monitoring and Debugging

### Check Data Parallel Status

```python
# Get information about the data parallel setup
info = encoder.get_data_parallel_info()
print(info)
# Output: {'enabled': True, 'device_count': 2, 'devices': ['cuda:0', 'cuda:1'], 'primary_device': 'cuda:0'}

# Check if data parallel is active
print(encoder.is_data_parallel)  # True/False
```

### Validation

The encoder automatically validates the data parallel setup and issues warnings for:
- CUDA not available
- Insufficient GPUs
- Invalid device IDs

```python
# Manual validation
issues = encoder.validate_data_parallel_setup()
if issues:
    print("Validation issues:", issues)
```

## Performance Tips

### Optimal Configuration

1. **Batch Size**: Use larger batch sizes (4-8x single GPU batch size)
2. **Memory**: Ensure sufficient GPU memory on all devices
3. **Workers**: Increase `preprocess_workers` for better CPU utilization
4. **Mixed Precision**: Enable `autocast=True` for memory savings

### Example Configuration

```python
encoder = get_encoder(
    model_name="prot_bert",
    device="cuda:0",
    batch_size=16,  # Large batch size
    data_parallel=True,
    device_ids=[0, 1, 2, 3],  # Use 4 GPUs
    autocast=True,  # Mixed precision
    preprocess_workers=4  # Parallel preprocessing
)
```

## Limitations and Considerations

### Current Limitations

- **Model compatibility**: Works with most models, but some custom models may need adaptation
- **Memory requirements**: All GPUs must have sufficient memory for the model
- **Synchronization overhead**: Small overhead for gradient synchronization (minimal for inference)

### Best Practices

1. **Test first**: Always test with a small dataset before large-scale processing
2. **Monitor memory**: Check GPU memory usage across all devices
3. **Batch size tuning**: Experiment with different batch sizes for optimal performance
4. **Error handling**: Implement proper error handling for multi-GPU scenarios

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use fewer GPUs
2. **Device not found**: Check `device_ids` are valid
3. **Performance degradation**: Ensure batch size is large enough to benefit from parallelization

### Debug Commands

```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

# Check encoder setup
print(f"Device: {encoder.device}")
print(f"Data parallel info: {encoder.get_data_parallel_info()}")
```

## Example Scripts

See `examples/data_parallel_example.py` for complete usage examples including:
- Basic data parallel setup
- Performance benchmarking
- Error handling
- Configuration validation

## Integration with Existing Code

The data parallel functionality is backward compatible. Existing code will continue to work without changes:

```python
# Existing code (single GPU)
encoder = get_encoder("prot_bert", device="cuda:0")

# Enhanced code (data parallel)
encoder = get_encoder("prot_bert", device="cuda:0", data_parallel=True)
```

The API remains the same - only the internal processing is distributed across multiple GPUs. 