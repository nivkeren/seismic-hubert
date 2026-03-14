"""
Seismic HuBERT: HuBERT architecture adapted for seismic waveforms.

HuBERT (Hidden Unit BERT) uses self-supervised learning with masked prediction
of discrete cluster assignments. This implementation adapts it for seismic data:
- Input: 100 Hz seismic waveforms (vs 16 kHz audio)
- Multi-channel support (E, N, Z components)
- Adjusted CNN feature extractor for lower sample rates
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    HubertModel,
    HubertConfig,
)
from typing import Optional
from dataclasses import dataclass


@dataclass
class SeismicHubertConfig:
    """Configuration for Seismic HuBERT model."""
    
    # Input specifications
    sample_rate: int = 100  # Seismic data sample rate (Hz)
    num_channels: int = 1  # Number of input channels (1 for Z, 3 for ENZ)
    waveform_length: int = 6000  # 60 seconds at 100 Hz
    
    # Transformer config
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # CNN Feature Extractor config
    # Adjusted for 100 Hz seismic data (vs 16 kHz audio)
    # Total stride = 32 → 6000 samples → ~187 frames (good for attention)
    conv_dim: tuple[int, ...] = (512, 512, 512, 512, 512)
    conv_stride: tuple[int, ...] = (2, 2, 2, 2, 2)  # Total stride: 32
    conv_kernel: tuple[int, ...] = (10, 8, 4, 4, 4)
    
    # HuBERT-specific
    num_clusters: int = 100  # K-means clusters for masked prediction
    mask_prob: float = 0.065  # ~6.5% of frames are mask start points
    mask_length: int = 5  # Consecutive frames to mask (~1.6 sec at 32x stride)
    
    # Pretrained model (optional - not recommended for seismic)
    pretrained_model: str | None = None
    
    @property
    def num_frames(self) -> int:
        """Approximate number of output frames for waveform_length input."""
        length = self.waveform_length
        for k, s in zip(self.conv_kernel, self.conv_stride):
            length = (length - k) // s + 1
        return length
    
    def to_hubert_config(self) -> HubertConfig:
        """Convert to HuggingFace HubertConfig."""
        return HubertConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_dropout=self.hidden_dropout,
            attention_probs_dropout_prob=self.attention_dropout,
            conv_dim=self.conv_dim,
            conv_stride=self.conv_stride,
            conv_kernel=self.conv_kernel,
            num_feat_extract_layers=len(self.conv_dim),
            feat_extract_activation="gelu",
            mask_time_prob=self.mask_prob,
            mask_time_length=self.mask_length,
        )
    
    @property 
    def total_stride(self) -> int:
        """Total downsampling factor of the conv encoder."""
        result = 1
        for s in self.conv_stride:
            result *= s
        return result
    
    def __repr__(self) -> str:
        return (
            f"SeismicHubertConfig(\n"
            f"  input: {self.waveform_length} samples @ {self.sample_rate}Hz, {self.num_channels} ch\n"
            f"  output: ~{self.num_frames} frames (total stride={self.total_stride})\n"
            f"  transformer: {self.num_hidden_layers} layers, {self.hidden_size} dim\n"
            f"  masking: {self.mask_prob:.1%} prob, {self.mask_length} consecutive\n"
            f")"
        )


class ChannelProjection(nn.Module):
    """Project multi-channel seismic input to single channel for HuBERT."""
    
    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.projection = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, samples) -> (batch, 1, samples)
        return self.projection(x)


class SeismicFeatureEncoder(nn.Module):
    """
    CNN feature encoder adapted for seismic waveforms.
    
    Seismic data at 100 Hz requires different downsampling than 16 kHz audio.
    This encoder produces ~1 feature vector per 3.2 seconds for 100 Hz input.
    """
    
    def __init__(self, config: SeismicHubertConfig):
        super().__init__()
        self.config = config
        
        self.channel_proj = None
        if config.num_channels > 1:
            self.channel_proj = ChannelProjection(config.num_channels, 1)
        
        conv_layers = []
        in_channels = 1
        
        for i, (out_channels, kernel_size, stride) in enumerate(
            zip(config.conv_dim, config.conv_kernel, config.conv_stride)
        ):
            conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    nn.GroupNorm(1, out_channels),  # Layer norm over channels
                    nn.GELU(),
                )
            )
            in_channels = out_channels
        
        self.conv_layers = nn.ModuleList(conv_layers)
        self.output_dim = config.conv_dim[-1]
    
    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input waveform of shape (batch, channels, samples)
        attention_mask : torch.Tensor, optional
            Mask of shape (batch, samples)
        
        Returns
        -------
        tuple
            Features of shape (batch, time, features) and output attention mask
        """
        if self.channel_proj is not None:
            x = self.channel_proj(x)
        
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dim if missing
        
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        x = x.transpose(1, 2)  # (batch, channels, time) -> (batch, time, channels)
        
        output_mask = None
        if attention_mask is not None:
            output_length = self._get_output_length(attention_mask.sum(dim=-1))
            max_len = x.shape[1]
            output_mask = torch.arange(max_len, device=x.device).expand(
                x.shape[0], -1
            ) < output_length.unsqueeze(1)
        
        return x, output_mask
    
    def _get_output_length(self, input_length: torch.Tensor) -> torch.Tensor:
        """Calculate output sequence length after convolutions."""
        for kernel, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_length = (input_length - kernel) // stride + 1
        return input_length


class SeismicHubert(nn.Module):
    """
    HuBERT model adapted for seismic waveform representation learning.
    
    This model can be used for:
    1. Self-supervised pretraining with masked prediction
    2. Fine-tuning for downstream tasks (phase picking, event detection, etc.)
    3. Feature extraction for clustering or classification
    """
    
    def __init__(self, config: SeismicHubertConfig):
        super().__init__()
        self.config = config
        
        self.feature_encoder = SeismicFeatureEncoder(config)
        
        # Feature projection to hidden size
        self.feature_projection = nn.Linear(
            config.conv_dim[-1], config.hidden_size
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_hidden_layers
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Parameters
        ----------
        input_values : torch.Tensor
            Input waveforms of shape (batch, channels, samples) or (batch, samples)
        attention_mask : torch.Tensor, optional
            Attention mask of shape (batch, samples)
        output_hidden_states : bool
            Whether to return all hidden states
        
        Returns
        -------
        dict
            Dictionary with 'last_hidden_state' and optionally 'hidden_states'
        """
        features, feature_mask = self.feature_encoder(input_values, attention_mask)
        
        features = self.feature_projection(features)
        features = self.layer_norm(features)
        features = self.dropout(features)
        
        src_key_padding_mask = None
        if feature_mask is not None:
            src_key_padding_mask = ~feature_mask
        
        hidden_states = self.encoder(
            features, src_key_padding_mask=src_key_padding_mask
        )
        
        return {
            "last_hidden_state": hidden_states,
            "feature_mask": feature_mask,
        }
    
    def extract_features(
        self, input_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract features without any task-specific head."""
        outputs = self.forward(input_values, attention_mask)
        return outputs["last_hidden_state"]


class SeismicHubertForPreTraining(nn.Module):
    """
    Seismic HuBERT with masked prediction head for self-supervised pretraining.
    
    The model learns to predict cluster assignments for masked time steps,
    similar to BERT's masked language modeling but for continuous signals.
    """
    
    def __init__(self, config: SeismicHubertConfig):
        super().__init__()
        self.config = config
        self.hubert = SeismicHubert(config)
        
        # Projection head for cluster prediction
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.final_proj = nn.Linear(config.hidden_size, config.num_clusters)
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with optional masked prediction loss.
        
        Parameters
        ----------
        input_values : torch.Tensor
            Input waveforms
        attention_mask : torch.Tensor, optional
            Attention mask
        labels : torch.Tensor, optional
            Cluster labels for masked positions (from K-means on features)
        mask_time_indices : torch.Tensor, optional
            Boolean mask indicating which time steps are masked
        
        Returns
        -------
        dict
            Dictionary with 'loss', 'logits', and 'hidden_states'
        """
        outputs = self.hubert(input_values, attention_mask)
        hidden_states = outputs["last_hidden_state"]
        
        projected = F.gelu(self.proj(hidden_states))
        logits = self.final_proj(projected)
        
        loss = None
        if labels is not None and mask_time_indices is not None:
            masked_logits = logits[mask_time_indices]
            masked_labels = labels[mask_time_indices]
            loss = F.cross_entropy(masked_logits, masked_labels)
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": hidden_states,
        }


def load_seismic_hubert(
    pretrained_model_name: str = "facebook/hubert-base-ls960",
    config: Optional[SeismicHubertConfig] = None,
    adapt_pretrained: bool = True,
) -> SeismicHubert:
    """
    Load a Seismic HuBERT model, optionally from a pretrained HuBERT.
    
    Parameters
    ----------
    pretrained_model_name : str
        HuggingFace model name for pretrained HuBERT
    config : SeismicHubertConfig, optional
        Custom configuration (uses defaults if not provided)
    adapt_pretrained : bool
        Whether to load pretrained weights and adapt them
    
    Returns
    -------
    SeismicHubert
        The model instance
    """
    if config is None:
        config = SeismicHubertConfig()
    
    model = SeismicHubert(config)
    
    if adapt_pretrained:
        print(f"Loading pretrained HuBERT from {pretrained_model_name}...")
        pretrained = HubertModel.from_pretrained(pretrained_model_name)
        
        # Transfer compatible weights from transformer encoder
        pretrained_state = pretrained.state_dict()
        model_state = model.state_dict()
        
        transferred = 0
        for name, param in pretrained_state.items():
            # Match encoder layers by shape
            if "encoder.layer" in name and name in model_state:
                if model_state[name].shape == param.shape:
                    model_state[name] = param
                    transferred += 1
        
        model.load_state_dict(model_state)
        print(f"Transferred {transferred} parameters from pretrained model")
    
    return model


if __name__ == "__main__":
    config = SeismicHubertConfig(num_channels=1)
    model = SeismicHubert(config)
    
    batch_size = 4
    num_samples = 6000  # 60 seconds at 100 Hz
    x = torch.randn(batch_size, 1, num_samples)
    
    print(f"Input shape: {x.shape}")
    outputs = model(x)
    print(f"Output shape: {outputs['last_hidden_state'].shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
