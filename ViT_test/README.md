# ViT-tiny Sensitivity Tests

This directory contains comprehensive sensitivity tests for ViT-tiny model deployment on CIM hardware.

## Directory Structure

```
ViT_test/
├── Quant_test/             # Quantization sensitivity tests
│   ├── by_layer_type/      # 8 layer types × 7 bit-widths = 56 tests
│   └── by_individual_layer/ # 96 layers × 7 bit-widths = 672 tests
└── Noise_test/             # Noise sensitivity tests
    ├── by_layer_type/      # 6 layer types × 5 noise levels = 30 tests
    └── by_individual_layer/ # 72 layers × 5 noise levels = 360 tests
```

## Test Categories

### 1. Quantization Sensitivity Tests (`Quant_test/`)

Tests how different layers respond to bit-width reduction (INT8 → INT2).

**Purpose**: Identify which layers need higher precision and which can tolerate lower precision.

**Key Findings**:
- Which layer types are most sensitive to quantization
- Which specific layers within each type show different sensitivities
- Optimal bit-width allocation for mixed-precision design

**Applications**:
- Mixed-precision configuration design
- Hardware resource allocation (higher precision = more area)
- Genetic algorithm initialization

### 2. Noise Sensitivity Tests (`Noise_test/`)

Tests how different layers respond to NVM noise at fixed INT8 quantization.

**Purpose**: Identify which layers need SRAM (low noise) and which can use NVM (higher noise, smaller area).

**Key Findings**:
- Which layer types are most sensitive to NVM noise
- Which specific layers can tolerate higher noise levels
- Optimal SRAM/NVM allocation for area optimization

**Applications**:
- SRAM vs NVM allocation decisions
- Noise-aware training (NAT) strategy
- Hardware design trade-offs (area vs accuracy)

## Test Design Philosophy

### Layer Exclusions

- **PatchEmbed and Head**: Always kept at FP32 (not tested)
  - Rationale: These are critical boundary layers with minimal area impact

- **QK and AV** (only in Noise_test): No weights, no noise injection
  - Rationale: Matrix multiplications without stored weights don't experience NVM noise

### Test Strategy

- **Isolated testing**: Only one layer (or layer type) is modified per test, all others remain at FP32
- **Baseline comparison**: All results compared against FP32 baseline accuracy
- **No confounding factors**: Quantization tests have no noise; noise tests use fixed INT8

## Quick Start Guide

### For Quick Overview (Layer Type Tests)

```bash
# Quantization sensitivity (56 tests, ~3-5 min)
cd Quant_test/by_layer_type
python test_layer_type_quant.py
python plot_quant_results.py

# Noise sensitivity (30 tests, ~2-3 min)
cd ../../Noise_test/by_layer_type
python test_layer_type_noise.py
python plot_noise_results.py
```

### For Detailed Analysis (Individual Layer Tests)

```bash
# Quantization sensitivity (672 tests, ~25-50 min)
cd Quant_test/by_individual_layer
python test_individual_layer_quant.py
python plot_individual_results.py

# Noise sensitivity (360 tests, ~15-30 min)
cd ../../Noise_test/by_individual_layer
python test_individual_layer_noise.py
python plot_individual_noise_results.py
```

## Interpreting Results

### Quantization Sensitivity

- **High sensitivity**: Large accuracy drop with lower bit-widths
  - → Needs higher precision (INT8/INT6)
  - → Allocate more SRAM bits

- **Low sensitivity**: Small accuracy drop even at INT2/INT3
  - → Can use lower precision (INT4/INT2)
  - → Save area with fewer bits

### Noise Sensitivity

- **High sensitivity**: Large accuracy drop with higher noise (σ=0.2, 0.25)
  - → Needs SRAM (low noise)
  - → Critical for accuracy preservation

- **Low sensitivity**: Small accuracy drop even at σ=0.25
  - → Can use NVM (higher noise)
  - → Save area (NVM is 10× smaller than SRAM)

### Combined Analysis

| Quant Sensitivity | Noise Sensitivity | Recommendation |
|-------------------|-------------------|----------------|
| High | High | High-precision SRAM (e.g., INT8, 0/8) |
| High | Low | High-precision NVM (e.g., INT8, 4/4) |
| Low | High | Low-precision SRAM (e.g., INT4, 0/4) |
| Low | Low | Low-precision NVM (e.g., INT4, 2/2) |

## Integration with Main Project

These test results inform:

1. **XGBoost Classification** (`analysis/layer_sensitivity.py`)
   - Use test results to validate/refine the 9-group classification
   - QH/QM/QL × NH/NM/NL sensitivity categories

2. **Genetic Algorithm** (`algorithms/coarse_ga.py`, `algorithms/fine_ga.py`)
   - Initialize population with sensitivity-aware configurations
   - Constrain search space based on sensitivity patterns

3. **Hardware Configuration** (`config/global_config.py`)
   - Define CONFIG_TABLE entries based on sensitivity requirements
   - Set GROUP_ALLOWED_CONFIGS based on test results

4. **Noise-Aware Training** (future work)
   - Focus NAT on noise-sensitive layers identified by tests
   - Skip NAT for noise-tolerant layers to save training time

## Visualization Outputs

Each test generates multiple plots:

- **Combined plots**: All layers/types on one graph
- **Individual plots**: Separate graph for each layer/type
- **Heatmaps**: Matrix view of all combinations
- **Rankings**: Bar charts showing most/least sensitive layers
- **Comparisons**: Side-by-side comparisons of different conditions

## Notes

- All tests use dummy accuracy when real dataset is unavailable
- Tests are designed to run on both CPU and GPU
- Results are saved in CSV format for further analysis
- Plots are saved at 300 DPI for publication quality

## Citation

If you use these tests in your research, please cite:
```
[Your paper citation here]
```
