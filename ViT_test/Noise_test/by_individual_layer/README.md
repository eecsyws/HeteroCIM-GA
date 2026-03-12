# Individual Layer Noise Sensitivity Test

This directory contains code to test the noise sensitivity of each individual layer (72 layers total) in ViT-tiny.

## Test Design

- **Total Layers**: 72 layers (12 blocks × 6 ops per block)
  - `blocks.{i}.attn.{q,k,v,proj}`: 12 × 4 = 48 layers
  - `blocks.{i}.mlp.{fc1,fc2}`: 12 × 2 = 24 layers
- **Excluded Layers**: PatchEmbed and Head (kept at FP32), QK and AV (no weights, no noise)
- **Quantization**: Fixed at INT8 with all 8 bits subject to noise
- **Noise Levels**: σ = 0.05, 0.1, 0.15, 0.2, 0.25
- **Test Strategy**: For each test, only one specific layer is quantized with noise injection, all others remain at FP32
- **Noise Model**: Bit-level noise injection on quantized weights (simulating NVM variation)

## Files

- `test_individual_layer_noise.py`: Main test script that evaluates accuracy for each layer × noise level combination
- `plot_individual_noise_results.py`: Visualization script that generates plots from test results
- `results/`: Output directory for CSV results and plots

## Usage

1. Run the test:
```bash
python test_individual_layer_noise.py
```

This will:
- Test 72 layers with 5 noise levels (360 total tests)
- Save results to `results/individual_layer_noise_results.csv`
- Progress: Shows [current/total] for each test

2. Generate visualizations:
```bash
python plot_individual_noise_results.py
```

This will create:
- `all_layers_noise_group_*.png`: Multiple plots showing subsets of layers (15 layers per plot)
- `sensitivity_heatmap_noise.png`: Heatmap showing all layers × all noise levels
- `accuracy_drop_ranking_noise.png`: Bar chart ranking layers by accuracy drop (σ=0.05 vs σ=0.25)
- `layer_comparison_sigma_*.png`: Line plots showing accuracy across all layers at each noise level
- `top_10_sensitive_layers_noise.png`: Top 10 most noise-sensitive layers
- `top_20_sensitive_layers_noise.png`: Top 20 most noise-sensitive layers

## Expected Output

The test will identify:
- Which specific layers are most sensitive to NVM noise
- How noise sensitivity varies across different blocks (early vs late blocks)
- Which layer types within the same category show different noise sensitivities
- Fine-grained guidance for SRAM vs NVM allocation decisions

## Note

This test is more comprehensive than the layer-type test and takes longer to run (360 tests vs 30 tests).
