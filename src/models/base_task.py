import torch.nn as nn
from src.models.seismic_hubert import SeismicHubert

class SeismicHubertTask(nn.Module):
    """
    Abstract base class for Seismic HuBERT downstream tasks.
    Provides common functionality such as freezing layers.
    
    Subclasses must define `self.hubert` as an instance of `SeismicHubert`.
    """
    
    def freeze_feature_encoder(self):
        """Freeze the CNN feature encoder of the base model."""
        if not hasattr(self, 'hubert'):
            raise AttributeError("Subclasses of SeismicHubertTask must define 'self.hubert'.")
            
        for param in self.hubert.feature_encoder.parameters():
            param.requires_grad = False

    def freeze_base_model(self):
        """Freeze the entire base model (CNN + Transformer)."""
        if not hasattr(self, 'hubert'):
            raise AttributeError("Subclasses of SeismicHubertTask must define 'self.hubert'.")
            
        for param in self.hubert.parameters():
            param.requires_grad = False
