import torch

def _calculate_f1_metrics(true_positives: torch.Tensor, predicted_positives: torch.Tensor, possible_positives: torch.Tensor, eps: float = 1e-7):
    """Common helper to compute precision, recall, and F1 score."""
    recall = true_positives / (possible_positives + eps)
    precision = true_positives / (predicted_positives + eps)
    f1 = 2 * ((precision * recall) / (precision + recall + eps))
    return precision, recall, f1


def calculate_eqt_metrics(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7):
    """
    Calculate precision, recall, and F1 score at the sample level, similar to EQTransformer.
    
    Parameters
    ----------
    y_pred : torch.Tensor
        Predicted label probabilities [batch_size, num_classes, seq_length]
    y_true : torch.Tensor
        True label probabilities [batch_size, num_classes, seq_length]
    threshold : float
        Probability threshold for a positive prediction
    eps : float
        Epsilon for numerical stability
        
    Returns
    -------
    tuple of torch.Tensor
        precision, recall, f1 for each class [num_classes]
    """
    y_pred_bin = (y_pred > threshold).float()
    y_true_bin = (y_true > threshold).float()
    
    # Sum over batch and sequence length dimensions to get metrics per class
    true_positives = (y_true_bin * y_pred_bin).sum(dim=(0, 2))
    possible_positives = y_true_bin.sum(dim=(0, 2))
    predicted_positives = y_pred_bin.sum(dim=(0, 2))
    
    return _calculate_f1_metrics(true_positives, predicted_positives, possible_positives, eps)


def get_pick_indices(y_probs: torch.Tensor, threshold: float = 0.3):
    """Helper to find the argmax peak for each class and whether it exceeds a threshold."""
    max_probs, max_idx = torch.max(y_probs, dim=-1)
    valid_picks = max_probs > threshold
    return max_idx, valid_picks


def calculate_phasenet_metrics(y_pred: torch.Tensor, y_true: torch.Tensor, tol_samples: int = 10, threshold: float = 0.3, eps: float = 1e-7):
    """
    Calculate PhaseNet style metrics: Pick is a TP if predicted peak is within tol_samples of true peak.
    """
    pred_idx, pred_valid = get_pick_indices(y_pred, threshold)
    true_idx, true_valid = get_pick_indices(y_true, threshold)
    
    # Calculate TP: both valid, and distance <= tol_samples
    distance = torch.abs(pred_idx - true_idx)
    tp = (pred_valid & true_valid & (distance <= tol_samples)).float().sum(dim=0)
    
    predicted_positives = pred_valid.float().sum(dim=0)
    possible_positives = true_valid.float().sum(dim=0)
    
    return _calculate_f1_metrics(tp, predicted_positives, possible_positives, eps)


def calculate_seislm_metrics(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.3):
    """
    Calculate SeisLM style continuous timing metric (Mean Absolute Error of the residual in samples).
    Returns the MAE per class for valid picks.
    """
    pred_idx, pred_valid = get_pick_indices(y_pred, threshold)
    true_idx, true_valid = get_pick_indices(y_true, threshold)
    
    # We only compute MAE where both are valid
    valid_mask = pred_valid & true_valid
    
    # MAE per class
    mae = torch.zeros(y_pred.shape[1], device=y_pred.device)
    for c in range(y_pred.shape[1]):
        mask_c = valid_mask[:, c]
        if mask_c.sum() > 0:
            mae[c] = torch.abs(pred_idx[mask_c, c] - true_idx[mask_c, c]).float().mean()
            
    return mae