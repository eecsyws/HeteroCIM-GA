# Noise Sensitivity Tests

This directory contains two types of noise sensitivity tests for ViT-tiny model.

## Directory Structure

```
Noise_test/
├── by_layer_type/          # Test by layer type (6 types)
│   ├── test_layer_type_noise.py
│   ├── plot_noise_results.py
│   ├── README.md
│   └── results/
└── by_individual_layer/    # Test by individual layer (72 layers)
    ├── test_individual_layer_noise.py
    ├── plot_individual_noise_results.py
    ├── README.md
    └── results/
```

## Test Types

### 1. By Layer Type (`by_layer_type/`)

Tests 6 layer types with different noise levels:
- **Layer Types**: QLinear, KLinear, VLinear, OutputLinear, FC1, FC2
- **Excluded**: PatchEmbed and Head (kept at FP32), QK and AV (no weights, no noise)
- **Total Tests**: 6 types × 5 noise levels = 30 tests
- **Use Case**: Quick overview of which layer types are sensitive to NVM noise

### 2. By Individual Layer (`by_individual_layer/`)

Tests each of the 72 layers individually:
- **Total Layers**: 72 (48 attention ops + 24 MLP ops)
- **Excluded**: PatchEmbed and Head (kept at FP32), QK and AV (no weights, no noise)
- **Total Tests**: 72 layers × 5 noise levels = 360 tests
- **Use Case**: Fine-grained analysis for precise SRAM/NVM allocation

## Common Settings

- **Quantization**: Fixed at INT8 with all 8 bits subject to noise
- **Noise Levels**: σ = 0.05, 0.1, 0.15, 0.2, 0.25
- **Test Strategy**: Only the target layer(s) are quantized with noise, all others remain at FP32
- **Noise Model**: Bit-level noise injection on quantized weights (simulating NVM variation)
- **Baseline**: FP32 model accuracy as reference

## Why Exclude QK and AV?

QK^T and AV are matrix multiplication operations without stored weights. Since NVM noise only affects stored weights in memory, these operations don't experience noise and are excluded from testing.

## Quick Start

1. **Run layer type test** (faster, ~2-3 minutes):
```bash
cd by_layer_type
python test_layer_type_noise.py
python plot_noise_results.py
```

2. **Run individual layer test** (comprehensive, ~15-30 minutes):
```bash
cd by_individual_layer
python test_individual_layer_noise.py
python plot_individual_noise_results.py
```

## Results Interpretation

- **High sensitivity**: Large accuracy drop with higher noise → needs SRAM (low noise)
- **Low sensitivity**: Small accuracy drop → can use NVM (higher noise but smaller area)
- **Layer type patterns**: Identify which operations (Q/K/V, MLP, etc.) are critical for noise tolerance
- **Block patterns**: Check if early/middle/late blocks have different noise sensitivities

## Applications

These test results guide:
1. **SRAM vs NVM allocation**: Allocate SRAM to noise-sensitive layers, NVM to noise-tolerant layers
2. **Hardware resource optimization**: Balance area savings vs accuracy degradation
3. **Noise-aware training (NAT)**: Identify which layers benefit most from NAT
4. **Genetic algorithm constraints**: Use noise sensitivity to constrain search space
5. **Architecture insights**: Understand which components are critical for noise robustness

## Relationship to Quantization Tests

- **Quantization tests** (`Quant_test/`): Test sensitivity to bit-width reduction (INT8 → INT2)
- **Noise tests** (`Noise_test/`): Test sensitivity to NVM noise at fixed INT8 quantization
- **Combined insight**: Layers sensitive to both quantization and noise need high-precision SRAM; layers tolerant to both can use low-precision NVM
