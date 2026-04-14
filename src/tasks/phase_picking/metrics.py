import torch
from torchmetrics import Metric

class PhasePickingMetrics(Metric):
    """
    PyTorch Lightning / torchmetrics compatible metrics for Phase Picking.
    Handles DDP sync and accumulation over epochs automatically.
    """
    def __init__(self, metric_type="eqt", tol_samples=10, threshold_eqt=0.5, threshold_pick=0.3, num_classes=3, **kwargs):
        super().__init__(**kwargs)
        self.metric_type = metric_type
        self.tol_samples = tol_samples
        self.threshold_eqt = threshold_eqt
        self.threshold_pick = threshold_pick
        self.num_classes = num_classes
        
        self.add_state("sum_err", default=torch.zeros(num_classes), dist_reduce_fx="sum")

        if metric_type in ["eqt", "phasenet"]:
            self.add_state("tp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
            self.add_state("pred_p", default=torch.zeros(num_classes), dist_reduce_fx="sum")
            self.add_state("true_p", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        elif metric_type == "seislm":
            self.add_state("count", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def _get_pick_indices(self, y_probs: torch.Tensor, threshold: float):
        max_probs, max_idx = torch.max(y_probs, dim=-1)
        valid_picks = max_probs > threshold
        return max_idx, valid_picks

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        if self.metric_type == "eqt":
            y_pred_bin = (y_pred > self.threshold_eqt).float()
            y_true_bin = (y_true > self.threshold_eqt).float()
            
            self.tp += (y_true_bin * y_pred_bin).sum(dim=(0, 2))
            self.true_p += y_true_bin.sum(dim=(0, 2))
            self.pred_p += y_pred_bin.sum(dim=(0, 2))
            
        elif self.metric_type == "phasenet":
            pred_idx, pred_valid = self._get_pick_indices(y_pred, self.threshold_pick)
            true_idx, true_valid = self._get_pick_indices(y_true, self.threshold_pick)
            
            distance = torch.abs(pred_idx - true_idx)
            tp_mask = pred_valid & true_valid & (distance <= self.tol_samples)
            self.tp += tp_mask.float().sum(dim=0)
            self.pred_p += pred_valid.float().sum(dim=0)
            self.true_p += true_valid.float().sum(dim=0)
            
            for c in range(self.num_classes):
                mask_c = tp_mask[:, c]
                if mask_c.sum() > 0:
                    self.sum_err[c] += distance[mask_c, c].float().sum()
            
        elif self.metric_type == "seislm":
            pred_idx, pred_valid = self._get_pick_indices(y_pred, self.threshold_pick)
            true_idx, true_valid = self._get_pick_indices(y_true, self.threshold_pick)
            
            valid_mask = pred_valid & true_valid
            for c in range(self.num_classes):
                mask_c = valid_mask[:, c]
                if mask_c.sum() > 0:
                    self.sum_err[c] += torch.abs(pred_idx[mask_c, c] - true_idx[mask_c, c]).float().sum()
                    self.count[c] += mask_c.sum().float()

    def compute(self):
        eps = 1e-7
        if self.metric_type in ["eqt", "phasenet"]:
            recall = self.tp / (self.true_p + eps)
            precision = self.tp / (self.pred_p + eps)
            f1 = 2 * ((precision * recall) / (precision + recall + eps))
            if self.metric_type == "phasenet":
                mae = self.sum_err / (self.tp + eps)
                return precision, recall, f1, mae
            return precision, recall, f1
        elif self.metric_type == "seislm":
            return self.sum_err / (self.count + eps)
