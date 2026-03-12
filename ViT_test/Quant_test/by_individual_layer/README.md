# Individual Layer Quantization Sensitivity Test

This directory contains code to test the quantization sensitivity of each individual layer (96 layers total) in ViT-tiny.

## Test Design

- **Total Layers**: 96 layers (12×8 transformer ops only)
- **Excluded Layers**: PatchEmbed and Head are kept at FP32 (not tested)
- **Quantization Levels**: INT8, INT7, INT6, INT5, INT4, INT3, INT2
- **Test Strategy**: For each test, only one specific layer is quantized while all others remain at FP32 precision
- **No Noise Injection**: Only quantization is applied, no NVM noise

## Layer Breakdown

- `blocks.{i}.attn.{q,k,v,qk,av,proj}`: 12 × 6 = 72 layers
- `blocks.{i}.mlp.{fc1,fc2}`: 12 × 2 = 24 layers

Total: 96 layers (PatchEmbed and Head excluded)

## Files

- `test_individual_layer_quant.py`: Main test script that evaluates accuracy for each layer × bit-width combination
- `plot_individual_results.py`: Visualization script that generates plots from test results
- `results/`: Output directory for CSV results and plots

## Usage

1. Run the test:
```bash
python test_individual_layer_quant.py
```

This will:
- Test 96 layers with 7 quantization levels (672 total tests)
- Save results to `results/individual_layer_quant_results.csv`
- Progress: Shows [current/total] for each test

2. Generate visualizations:
```bash
python plot_individual_results.py
```

This will create:
- `all_layers_group_*.png`: Multiple plots showing subsets of layers (20 layers per plot)
- `sensitivity_heatmap.png`: Heatmap showing all layers × all bit-widths
- `accuracy_drop_ranking.png`: Bar chart ranking layers by accuracy drop (INT8 vs INT2)
- `layer_comparison_INT*.png`: Line plots showing accuracy across all layers at each bit-width
- `top_10_sensitive_layers.png`: Top 10 most sensitive layers
- `top_20_sensitive_layers.png`: Top 20 most sensitive layers

## Expected Output

The test will identify:
- Which specific layers are most sensitive to quantization
- How sensitivity varies across different blocks (early vs late blocks)
- Which layer types within the same category show different sensitivities
- Fine-grained guidance for mixed-precision configuration design

## Note

This test is more comprehensive than the layer-type test and takes longer to run (672 tests vs 56 tests).
