# Quantization Sensitivity Tests

This directory contains two types of quantization sensitivity tests for ViT-tiny model.

## Directory Structure

```
Quant_test/
├── by_layer_type/          # Test by layer type (8 types)
│   ├── test_layer_type_quant.py
│   ├── plot_quant_results.py
│   ├── README.md
│   └── results/
└── by_individual_layer/    # Test by individual layer (96 layers)
    ├── test_individual_layer_quant.py
    ├── plot_individual_results.py
    ├── README.md
    └── results/
```

## Test Types

### 1. By Layer Type (`by_layer_type/`)

Tests 8 layer types with different quantization levels:
- **Layer Types**: QLinear, KLinear, VLinear, QK^T, AV, OutputLinear, FC1, FC2
- **Excluded**: PatchEmbed and Head (kept at FP32)
- **Total Tests**: 8 types × 7 bit-widths = 56 tests
- **Use Case**: Quick overview of which layer types are sensitive to quantization

### 2. By Individual Layer (`by_individual_layer/`)

Tests each of the 96 layers individually:
- **Total Layers**: 96 (72 attention ops + 24 MLP ops)
- **Excluded**: PatchEmbed and Head (kept at FP32)
- **Total Tests**: 96 layers × 7 bit-widths = 672 tests
- **Use Case**: Fine-grained analysis for precise mixed-precision configuration

## Common Settings

- **Quantization Levels**: INT8, INT7, INT6, INT5, INT4, INT3, INT2
- **Test Strategy**: Only the target layer(s) are quantized, all others remain at FP32
- **No Noise Injection**: Pure quantization effect without NVM noise
- **Baseline**: FP32 model accuracy as reference

## Quick Start

1. **Run layer type test** (faster, ~3-5 minutes):
```bash
cd by_layer_type
python test_layer_type_quant.py
python plot_quant_results.py
```

2. **Run individual layer test** (comprehensive, ~25-50 minutes):
```bash
cd by_individual_layer
python test_individual_layer_quant.py
python plot_individual_results.py
```

## Results Interpretation

- **High sensitivity**: Large accuracy drop with lower bit-widths → needs higher precision
- **Low sensitivity**: Small accuracy drop → can use lower precision to save area
- **Layer type patterns**: Identify which operations (Q/K/V, MLP, etc.) are critical
- **Block patterns**: Check if early/middle/late blocks have different sensitivities

## Applications

These test results guide:
1. **Mixed-precision design**: Allocate higher precision to sensitive layers
2. **Hardware resource allocation**: Balance SRAM/NVM usage based on sensitivity
3. **Genetic algorithm initialization**: Use sensitivity as prior knowledge
4. **Architecture insights**: Understand which components are critical for accuracy
