# Phase Picking Fine-Tuning

After pretraining the Seismic HuBERT foundation model, you can fine-tune it for specific downstream tasks. This guide details the methodology for **Phase Picking** (detecting P and S wave arrivals).

We provide an implementation for fine-tuning the model specifically for phase picking. The implementation supports two architectural approaches:

1. **Linear Probe (`SeismicHubertForPhasePicking`)**: The default and most scientifically rigorous method. It interpolates the HuBERT features to the original sample resolution and passes them through a single Linear layer. This proves the base features alone contain high-quality timing information without relying on complex downstream heads.
2. **SeisLM Approach (`SeismicHubertForPhasePickingSeisLM`)**: A more complex head that concatenates the upsampled features with the raw waveform and processes them through multiple CNN layers before classification.

## Running Phase Picking Fine-Tuning

You can train the phase picker from scratch, or (recommended) initialize it with your pretrained weights:

```bash
# Train from scratch (baseline)
pixi run python src/train_phase_picking.py

# Fine-tune using features learned from a pretraining run
pixi run python src/train_phase_picking.py pretrained_weights="outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/best.ckpt"
```

## Freezing Strategies

When fine-tuning from pretrained weights, you might want to freeze parts of the network to prevent catastrophic forgetting or to perform a strict linear probe evaluation:

```bash
# Freeze only the CNN feature encoder, fine-tune Transformer + Classifier
pixi run python src/train_phase_picking.py pretrained_weights="..." +freeze_feature_encoder=true

# Freeze the entire base model (Strict Linear Probe / frozen features)
pixi run python src/train_phase_picking.py pretrained_weights="..." +freeze_base_model=true
```