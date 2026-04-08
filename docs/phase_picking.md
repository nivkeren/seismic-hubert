# Phase Picking Fine-Tuning

After pretraining the Seismic HuBERT foundation model, you can fine-tune it for specific downstream tasks. This guide details the methodology for **Phase Picking** (detecting P and S wave arrivals).

We provide an implementation for fine-tuning the model specifically for phase picking. The implementation supports two architectural approaches:

1. **Linear Probe (`SeismicHubertForPhasePicking`)**: The default and most scientifically rigorous method. It interpolates the HuBERT features to the original sample resolution and passes them through a single Linear layer. This proves the base features alone contain high-quality timing information without relying on complex downstream heads.
2. **SeisLM Approach (`SeismicHubertForPhasePickingSeisLM`)**: A more complex head that concatenates the upsampled features with the raw waveform and processes them through multiple CNN layers before classification.

## Running Phase Picking Fine-Tuning

You can train the phase picker from scratch, or (recommended) initialize it with your pretrained weights:

```bash
# Train from scratch (baseline)
pixi run python src/tasks/phase_picking/train.py

# Fine-tune using features learned from a pretraining run
pixi run python src/tasks/phase_picking/train.py pretrained_weights="outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/best.ckpt"
```

## Freezing Strategies

When fine-tuning from pretrained weights, you might want to freeze parts of the network to prevent catastrophic forgetting or to perform a strict linear probe evaluation:

```bash
# Freeze only the CNN feature encoder, fine-tune Transformer + Classifier
pixi run python src/tasks/phase_picking/train.py pretrained_weights="..." +freeze_feature_encoder=true

# Freeze the entire base model (Strict Linear Probe / frozen features)
pixi run python src/tasks/phase_picking/train.py pretrained_weights="..." +freeze_base_model=true
```

## Metric Calculation References

Evaluating downstream tasks like phase picking involves defining what constitutes a True Positive (TP). Different seismic deep learning models define and calculate these metrics differently. Our implementation calculates metrics at the sample (or pixel) level during training, similar to EQTransformer.

Here are the reference implementations from three major models that influenced our metrics:

### 1. PhaseNet
**True Positive Definition:** A pick is considered a TP if the predicted peak is within a specified time tolerance (e.g., 0.1s) of the manual/true pick.

*From `phasenet/util.py`:*
```python
def correct_picks(picks, itp, its, tol):
    TP_p = 0; TP_s = 0; nP_p = 0; nP_s = 0; nT_p = 0; nT_s = 0
    # ... (peak finding and matching logic) ...
    diff_p = picks_p - itp_i
    TP_p += np.sum(np.abs(tmp_p) < tol/dt)
    
def calc_metrics(nTP, nP, nT):
    precision = nTP / nP
    recall = nTP / nT
    f1 = 2 * precision * recall / (precision + recall)
    return [precision, recall, f1]
```

### 2. EQTransformer
**True Positive Definition:** Calculated at the sample (or pixel) level during training. The Keras backend directly computes the dot product overlap between the predicted probability curve (rounded > 0.5) and the true label curve (which is a Gaussian or box distribution). 

*From `EQTransformer/core/EqT_utils.py`:*
```python
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))
```

### 3. SeisLM (pick-benchmark)
**True Positive Definition:** For timing accuracy, SeisLM does not use a binary threshold but instead calculates the continuous timing residual (Mean Absolute Error) between the predicted peak and the true onset. For pure classification (e.g. signal vs noise), it uses Scikit-Learn's `precision_recall_curve` to find the optimal F1 score threshold over the whole dataset.

*From `seisLM/evaluation/pick_eval.py`:*
```python
def get_results_onset_determination(pred_path):
    pred = pd.read_csv(pred_path)
    results = {}
    for phase in ['P', 'S']:
        pred_phase = pred[pred["phase_label"] == phase]
        pred_col = f"{phase.lower()}_sample_pred"
        diff = (pred_phase[pred_col] - pred_phase["phase_onset"]) / pred_phase["sampling_rate"]
        results[f'{phase}_onset_diff'] = diff
    return results

def get_results_phase_identification(pred_path):
    # ...
    prec, recall, thr = metrics.precision_recall_curve(pred["phase_label_bin"], pred["score_p_or_s"])
    f1 = 2 * prec * recall / (prec + recall)
    best_f1 = np.nanmax(f1)
    # ...
```