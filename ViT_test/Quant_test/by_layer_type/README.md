# Layer Type Quantization Sensitivity Test

This directory contains code to test the quantization sensitivity of different layer types in ViT-tiny.

## Test Design

- **Layer Types**: 8 types (QLinear, KLinear, VLinear, QK^T, AV, OutputLinear, FC1, FC2)
- **Excluded Layers**: PatchEmbed and Head are kept at FP32 (not tested)
- **Total Layers Tested**: 8 types × 12 blocks = 96 layers
- **Quantization Levels**: INT8, INT7, INT6, INT5, INT4, INT3, INT2
- **Test Strategy**: For each test, only one layer type is quantized while all others remain at FP32 precision
- **No Noise Injection**: Only quantization is applied, no NVM noise

## Files

- `test_layer_type_quant.py`: Main test script that evaluates accuracy for each layer type × bit-width combination
- `plot_quant_results.py`: Visualization script that generates plots from test results
- `results/`: Output directory for CSV results and plots

## Usage

1. Run the test:
```bash
python test_layer_type_quant.py
```

This will:
- Test 8 layer types with 7 quantization levels (56 total tests)
- Save results to `results/layer_type_quant_results.csv`

2. Generate visualizations:
```bash
python plot_quant_results.py
```

This will create:
- `all_layer_types.png`: Combined plot showing all layer types
- Individual plots for each layer type
- `accuracy_drop_heatmap.png`: Heatmap showing accuracy drop relative to INT8
- `sensitivity_ranking.png`: Bar chart ranking layer types by INT2 accuracy

## Expected Output

The test will identify which layer types are most sensitive to quantization, helping to guide the design of mixed-precision configurations.
