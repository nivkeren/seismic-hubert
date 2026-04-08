import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import pytorch_lightning as pl

from models.seismic_hubert import SeismicHubert, SeismicHubertConfig
from tasks.base_task import SeismicHubertTask


class DoubleConvBlock(nn.Module):
    """
    Two layers of 1D Convolution -> GroupNorm -> GELU -> Dropout.
    Used for the phase picking head to process concatenated features and inputs.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3, 
        dropout_rate: float = 0.1, 
        padding: str = "same"
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.drop1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.drop2(x)
        return x


class SeismicHubertForPhasePickingSeisLM(SeismicHubertTask):
    """
    Seismic HuBERT with a phase picking head, following the approach in SeisLM.
    
    The output of the base model is interpolated back to the original waveform resolution,
    concatenated with the original waveform itself, and passed through a DoubleConvBlock 
    and a linear classifier to produce per-sample predictions.
    """
    def __init__(
        self, 
        config: SeismicHubertConfig, 
        num_classes: int = 3, 
        head_dropout_rate: float = 0.1
    ):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        
        # Base model (can be loaded with pretrained weights later)
        self.hubert = SeismicHubert(config)
        
        # The input to DoubleConv is the concatenation of the original waveform and the upsampled hidden states
        in_channels = config.hidden_size + config.num_channels
        
        self.double_conv = DoubleConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            dropout_rate=head_dropout_rate,
            padding="same"
        )
        
        self.hidden_dropout = nn.Dropout(head_dropout_rate)
        self.classifier = nn.Linear(in_channels, num_classes)
        
    def forward(
        self, 
        input_values: torch.Tensor, 
        attention_mask: torch.Tensor = None
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        input_values: (batch, num_channels, seq_len) or (batch, seq_len)
        attention_mask: (batch, seq_len)
        
        Returns
        -------
        dict with:
            logits: (batch, num_classes, seq_len)
            probs: (batch, num_classes, seq_len)
        """
        if input_values.dim() == 2:
            input_values = input_values.unsqueeze(1)
            
        seq_len = input_values.shape[-1]
            
        # Get base model features
        outputs = self.hubert(input_values, attention_mask=attention_mask)
        hidden_states = outputs["last_hidden_state"]  # (batch, frames, hidden_size)
        
        # Interpolate hidden states to match original sequence length
        hidden_states = einops.rearrange(hidden_states, "b f h -> b h f")
        hidden_states = F.interpolate(hidden_states, size=seq_len, mode="linear", align_corners=False)
        # hidden_states: (batch, hidden_size, seq_len)
        
        # Concatenate original input and upsampled hidden states
        # input_values: (batch, num_channels, seq_len)
        combined = torch.cat([hidden_states, input_values], dim=1)
        
        # Apply DoubleConvBlock
        combined = self.double_conv(combined)
        
        # Rearrange for classifier
        combined = einops.rearrange(combined, "b c l -> b l c")
        combined = self.hidden_dropout(combined)
        
        # Classifier logits
        logits = self.classifier(combined)  # (batch, seq_len, num_classes)
        
        # Return logits in shape (batch, num_classes, seq_len) to match phase picking targets typically
        logits = einops.rearrange(logits, "b l c -> b c l")
        
        # Softmax over classes dimension
        probs = F.softmax(logits, dim=1)
        
        return {
            "logits": logits,
            "probs": probs
        }


class SeismicHubertForPhasePicking(SeismicHubertTask):
    """
    Seismic HuBERT with a linear probe phase picking head.
    
    The output of the base model is interpolated back to the original waveform resolution,
    and passed directly through a linear classifier to produce per-sample predictions.
    This rigorous setup proves that the base model learns high-quality, general representations.
    """
    def __init__(
        self, 
        config: SeismicHubertConfig, 
        num_classes: int = 3, 
    ):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        
        # Base model (can be loaded with pretrained weights later)
        self.hubert = SeismicHubert(config)
        
        # Linear probe classifier mapping hidden_size -> num_classes
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        
    def forward(
        self, 
        input_values: torch.Tensor, 
        attention_mask: torch.Tensor = None
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        input_values: (batch, num_channels, seq_len) or (batch, seq_len)
        attention_mask: (batch, seq_len)
        
        Returns
        -------
        dict with:
            logits: (batch, num_classes, seq_len)
            probs: (batch, num_classes, seq_len)
        """
        if input_values.dim() == 2:
            input_values = input_values.unsqueeze(1)
            
        seq_len = input_values.shape[-1]
            
        # Get base model features
        outputs = self.hubert(input_values, attention_mask=attention_mask)
        hidden_states = outputs["last_hidden_state"]  # (batch, frames, hidden_size)
        
        # Interpolate hidden states to match original sequence length
        hidden_states = einops.rearrange(hidden_states, "b f h -> b h f")
        hidden_states = F.interpolate(hidden_states, size=seq_len, mode="linear", align_corners=False)
        # hidden_states: (batch, hidden_size, seq_len)
        
        # Rearrange for classifier
        features = einops.rearrange(hidden_states, "b h l -> b l h")
        
        # Classifier logits
        logits = self.classifier(features)  # (batch, seq_len, num_classes)
        
        # Return logits in shape (batch, num_classes, seq_len)
        logits = einops.rearrange(logits, "b l c -> b c l")
        
        # Softmax over classes dimension
        probs = F.softmax(logits, dim=1)
        
        return {
            "logits": logits,
            "probs": probs
        }


from tasks.phase_picking.losses import vector_cross_entropy
from tasks.phase_picking.metrics import (
    calculate_eqt_metrics,
    calculate_phasenet_metrics,
    calculate_seislm_metrics
)


class PhasePickingLightning(pl.LightningModule):
    """
    PyTorch Lightning module for training the phase picking model.
    """
    def __init__(
        self,
        config: SeismicHubertConfig,
        num_classes: int = 3,
        head_dropout_rate: float = 0.1,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        freeze_feature_encoder: bool = False,
        freeze_base_model: bool = False,
        eval_metric: str = "eqt",  # "eqt", "phasenet", "seislm", "all"
        tolerance_samples: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eval_metric = eval_metric
        self.tolerance_samples = tolerance_samples
        
        self.model = SeismicHubertForPhasePicking(
            config=config,
            num_classes=num_classes,
            head_dropout_rate=head_dropout_rate
        )
        
        if freeze_feature_encoder:
            self.model.freeze_feature_encoder()
            
        if freeze_base_model:
            self.model.freeze_base_model()
            
    def forward(self, input_values, attention_mask=None):
        return self.model(input_values, attention_mask)
        
    def _shared_step(self, batch, batch_idx):
        # Adjust these keys based on your dataloader for the downstream task
        x = batch["input_values"]
        y_true = batch["labels"]  # Expected to be probabilities of shape [batch, num_classes, seq_len]
        attention_mask = batch.get("attention_mask", None)
        
        outputs = self(x, attention_mask)
        y_pred = outputs["probs"]
        
        loss = vector_cross_entropy(y_pred, y_true)
        return loss, y_pred, y_true

    def _log_metrics(self, y_pred, y_true, prefix="train"):
        if self.eval_metric in ["eqt", "all"]:
            precision, recall, f1 = calculate_eqt_metrics(y_pred, y_true)
            if precision.shape[0] >= 3:
                self.log(f"{prefix}_eqt_p_precision", precision[1], on_step=False, on_epoch=True, sync_dist=True)
                self.log(f"{prefix}_eqt_p_recall", recall[1], on_step=False, on_epoch=True, sync_dist=True)
                self.log(f"{prefix}_eqt_p_f1", f1[1], on_step=False, on_epoch=True, sync_dist=True)
                self.log(f"{prefix}_eqt_s_precision", precision[2], on_step=False, on_epoch=True, sync_dist=True)
                self.log(f"{prefix}_eqt_s_recall", recall[2], on_step=False, on_epoch=True, sync_dist=True)
                self.log(f"{prefix}_eqt_s_f1", f1[2], on_step=False, on_epoch=True, sync_dist=True)
                
        if self.eval_metric in ["phasenet", "all"]:
            precision, recall, f1 = calculate_phasenet_metrics(y_pred, y_true, tol_samples=self.tolerance_samples)
            if precision.shape[0] >= 3:
                self.log(f"{prefix}_phasenet_p_precision", precision[1], on_step=False, on_epoch=True, sync_dist=True)
                self.log(f"{prefix}_phasenet_p_recall", recall[1], on_step=False, on_epoch=True, sync_dist=True)
                self.log(f"{prefix}_phasenet_p_f1", f1[1], on_step=False, on_epoch=True, sync_dist=True)
                self.log(f"{prefix}_phasenet_s_precision", precision[2], on_step=False, on_epoch=True, sync_dist=True)
                self.log(f"{prefix}_phasenet_s_recall", recall[2], on_step=False, on_epoch=True, sync_dist=True)
                self.log(f"{prefix}_phasenet_s_f1", f1[2], on_step=False, on_epoch=True, sync_dist=True)
                
        if self.eval_metric in ["seislm", "all"]:
            mae = calculate_seislm_metrics(y_pred, y_true)
            if mae.shape[0] >= 3:
                self.log(f"{prefix}_seislm_p_mae", mae[1], on_step=False, on_epoch=True, sync_dist=True)
                self.log(f"{prefix}_seislm_s_mae", mae[2], on_step=False, on_epoch=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        loss, y_pred, y_true = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self._log_metrics(y_pred, y_true, prefix="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_pred, y_true = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self._log_metrics(y_pred, y_true, prefix="val")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = float(current_step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
