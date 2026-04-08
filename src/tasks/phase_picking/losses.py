import torch

def vector_cross_entropy(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Cross entropy loss for probability vectors.
    
    Parameters
    ----------
    y_true : torch.Tensor
        True label probabilities [batch_size, num_classes, seq_length]
    y_pred : torch.Tensor
        Predicted label probabilities [batch_size, num_classes, seq_length]
    eps : float
        Epsilon to clip values for stability
        
    Returns
    -------
    torch.Tensor
        Average loss across batch
    """
    h = y_true * torch.log(y_pred + eps)
    if y_pred.ndim == 3:
        # Mean along sample dimension and sum along class dimension
        h = h.mean(-1).sum(-1)
    else:
        h = h.sum(-1)  # Sum along class dimension
    return -h.mean()  # Mean over batch axis
