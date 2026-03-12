# Layer Type Noise Sensitivity Test

This directory contains code to test the noise sensitivity of different layer types in ViT-tiny.

## Test Design

- **Layer Types**: 6 types (QLinear, KLinear, VLinear, OutputLinear, FC1, FC2)
- **Excluded Layers**: PatchEmbed and Head (kept at FP32), QK and AV (no weights, no noise)
- **Total Layers Tested**: 6 types × 12 blocks = 72 layers
- **Quantization**: Fixed at INT8 with all 8 bits subject to noise
- **Noise Levels**: σ = 0.05, 0.1, 0.15, 0.2, 0.25
- **Test Strategy**: For each test, only one layer type is quantized with noise injection, all others remain at FP32
- **Noise Model**: Bit-level noise injection on quantized weights (simulating NVM variation)

## Files

- `test_layer_type_noise.py`: Main test script that evaluates accuracy for each layer type × noise level combination
- `plot_noise_results.py`: Visualization script that generates plots from test results
- `results/`: Output directory for CSV results and plots

## Usage

1. Run the test:
```bash
python test_layer_type_noise.py
```

This will:
- Test 6 layer types with 5 noise levels (30 total tests)
- Save results to `results/layer_type_noise_results.csv`

2. Generate visualizations:
```bash
python plot_noise_results.py
```

This will create:
- `all_layer_types_noise.png`: Combined plot showing all layer types
- Individual plots for each layer type
- `accuracy_drop_heatmap_noise.png`: Heatmap showing accuracy drop relative to σ=0.05
- `sensitivity_ranking_noise.png`: Bar chart ranking layer types by accuracy at σ=0.25
- `noise_impact_comparison.png`: Comparison of low noise (σ=0.05) vs high noise (σ=0.25)

## Expected Output

The test will identify which layer types are most sensitive to NVM noise, helping to guide:
- SRAM vs NVM allocation decisions
- Noise-aware training strategies
- Hardware design trade-offs

## Note on QK and AV

QK^T and AV operations are matrix multiplications without stored weights, so they don't experience NVM noise. They are excluded from this test.
