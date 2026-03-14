"""Tests for clustering and feature extraction."""

import numpy as np
import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.clustering import (
    extract_spectrogram_features,
    extract_stalta_features,
    extract_frequency_band_features,
    extract_multichannel_features,
    extract_combined_features,
    ClusterLabelGenerator,
)


class TestSpectrogramFeatures:
    """Tests for spectrogram feature extraction."""
    
    def test_1d_input(self):
        """Test with single-channel 1D input."""
        waveform = np.random.randn(6000)
        features = extract_spectrogram_features(waveform)
        
        assert features.ndim == 2
        assert features.shape[0] > 0  # Has frames
        assert features.shape[1] > 0  # Has features
        assert features.dtype == np.float32
    
    def test_2d_input_single_channel(self):
        """Test with single-channel 2D input."""
        waveform = np.random.randn(1, 6000)
        features = extract_spectrogram_features(waveform)
        
        assert features.ndim == 2
        assert features.shape[0] > 0
    
    def test_2d_input_multi_channel(self):
        """Test with 3-channel 2D input."""
        waveform = np.random.randn(3, 6000)
        features = extract_spectrogram_features(waveform)
        
        # Should use first channel only
        assert features.ndim == 2
        assert features.shape[0] > 0
    
    def test_custom_parameters(self):
        """Test with custom FFT parameters."""
        waveform = np.random.randn(6000)
        features = extract_spectrogram_features(
            waveform,
            sample_rate=100,
            n_fft=128,
            hop_length=64,
            n_mels=20,
        )
        
        assert features.shape[1] == 20  # n_mels frequency bins


class TestSTALTAFeatures:
    """Tests for STA/LTA feature extraction."""
    
    def test_1d_input(self):
        """Test with single-channel input."""
        waveform = np.random.randn(6000)
        features = extract_stalta_features(waveform)
        
        assert features.ndim == 2
        assert features.shape[0] > 0
        assert features.shape[1] == 2  # STA/LTA ratio + energy for 1 channel
    
    def test_3_channel_input(self):
        """Test with 3-channel input."""
        waveform = np.random.randn(3, 6000)
        features = extract_stalta_features(waveform)
        
        assert features.ndim == 2
        assert features.shape[1] == 6  # 2 features per channel * 3 channels
    
    def test_detects_transient(self):
        """Test that STA/LTA detects a transient signal."""
        # Create waveform with a spike
        waveform = np.zeros(6000)
        waveform[3000:3100] = 10.0  # Spike in the middle
        
        features = extract_stalta_features(waveform, hop_length=10)
        stalta_ratio = features[:, 0]
        
        # STA/LTA should be higher around the spike
        spike_region = 3000 // 10
        assert stalta_ratio[spike_region] > stalta_ratio[0]


class TestFrequencyBandFeatures:
    """Tests for frequency band feature extraction."""
    
    def test_1d_input(self):
        """Test with single-channel input."""
        waveform = np.random.randn(6000)
        features = extract_frequency_band_features(waveform)
        
        assert features.ndim == 2
        assert features.shape[0] > 0
        # Default: 4 bands * 1 channel = 4 features
        assert features.shape[1] == 4
    
    def test_3_channel_input(self):
        """Test with 3-channel input."""
        waveform = np.random.randn(3, 6000)
        features = extract_frequency_band_features(waveform)
        
        # Default: 4 bands * 3 channels = 12 features
        assert features.shape[1] == 12
    
    def test_custom_bands(self):
        """Test with custom frequency bands."""
        waveform = np.random.randn(6000)
        bands = [(1.0, 5.0), (5.0, 10.0)]
        features = extract_frequency_band_features(waveform, bands=bands)
        
        assert features.shape[1] == 2  # 2 custom bands


class TestMultichannelFeatures:
    """Tests for multi-channel polarization features."""
    
    def test_1d_input_returns_zeros(self):
        """Test that 1D input returns zero features."""
        waveform = np.random.randn(6000)
        features = extract_multichannel_features(waveform)
        
        assert features.ndim == 2
        assert features.shape[1] == 6  # 6 polarization features
        assert np.allclose(features, 0)  # All zeros for 1D input
    
    def test_3_channel_input(self):
        """Test with 3-channel input."""
        waveform = np.random.randn(3, 6000)
        features = extract_multichannel_features(waveform)
        
        assert features.ndim == 2
        assert features.shape[1] == 6  # 6 polarization features
        assert not np.allclose(features, 0)  # Should have non-zero values
    
    def test_vertical_motion_hv_ratio(self):
        """Test H/V ratio for vertical-dominant motion."""
        # Create Z-dominant waveform (like P-wave)
        waveform = np.zeros((3, 6000))
        waveform[2, :] = np.random.randn(6000)  # Z channel only
        
        features = extract_multichannel_features(waveform)
        hv_ratio = features[:, 0]
        
        # H/V ratio should be negative (log scale) when Z dominates
        assert np.mean(hv_ratio) < 0


class TestCombinedFeatures:
    """Tests for combined feature extraction."""
    
    def test_spectrogram_only(self):
        """Test with only spectrogram features."""
        waveform = np.random.randn(3, 6000)
        features = extract_combined_features(
            waveform,
            include_spectrogram=True,
            include_stalta=False,
            include_frequency_bands=False,
            include_multichannel=False,
        )
        
        assert features.ndim == 2
        assert features.shape[0] > 0
    
    def test_all_features(self):
        """Test with all feature types enabled."""
        waveform = np.random.randn(3, 6000)
        features = extract_combined_features(
            waveform,
            include_spectrogram=True,
            include_stalta=True,
            include_frequency_bands=True,
            include_multichannel=True,
        )
        
        assert features.ndim == 2
        # Should have more features than spectrogram alone
        spec_only = extract_combined_features(
            waveform,
            include_spectrogram=True,
            include_stalta=False,
            include_frequency_bands=False,
            include_multichannel=False,
        )
        assert features.shape[1] > spec_only.shape[1]
    
    def test_no_features_raises(self):
        """Test that disabling all features raises an error."""
        waveform = np.random.randn(6000)
        with pytest.raises(ValueError):
            extract_combined_features(
                waveform,
                include_spectrogram=False,
                include_stalta=False,
                include_frequency_bands=False,
                include_multichannel=False,
            )


class TestClusterLabelGenerator:
    """Tests for ClusterLabelGenerator."""
    
    def test_extract_features_spectrogram(self):
        """Test feature extraction with spectrogram mode."""
        generator = ClusterLabelGenerator(
            n_clusters=10,
            feature_mode="spectrogram",
        )
        
        waveform = np.random.randn(3, 6000)
        features = generator.extract_features(waveform)
        
        assert features.ndim == 2
        assert features.shape[0] > 0
    
    def test_extract_features_stalta(self):
        """Test feature extraction with STA/LTA mode."""
        generator = ClusterLabelGenerator(
            n_clusters=10,
            feature_mode="stalta",
        )
        
        waveform = np.random.randn(3, 6000)
        features = generator.extract_features(waveform)
        
        assert features.ndim == 2
    
    def test_extract_features_combined(self):
        """Test feature extraction with combined mode."""
        generator = ClusterLabelGenerator(
            n_clusters=10,
            feature_mode="combined",
            include_stalta=True,
            include_frequency_bands=True,
        )
        
        waveform = np.random.randn(3, 6000)
        features = generator.extract_features(waveform)
        
        assert features.ndim == 2
    
    def test_get_labels_not_fitted_raises(self):
        """Test that get_labels raises before fitting."""
        generator = ClusterLabelGenerator(n_clusters=10)
        waveform = np.random.randn(6000)
        
        with pytest.raises(RuntimeError):
            generator.get_labels(waveform)
    
    def test_get_labels_single_waveform(self):
        """Test get_labels with a single waveform after manual fitting."""
        generator = ClusterLabelGenerator(n_clusters=10, feature_dim=16)
        
        # Manually fit with random data
        features = np.random.randn(1000, 16).astype(np.float32)
        from sklearn.cluster import MiniBatchKMeans
        generator.kmeans = MiniBatchKMeans(n_clusters=10, random_state=42)
        generator.kmeans.fit(features)
        generator._fitted = True
        
        waveform = np.random.randn(6000)
        labels = generator.get_labels(waveform)
        
        assert isinstance(labels, torch.Tensor)
        assert labels.ndim == 1
        assert labels.min() >= 0
        assert labels.max() < 10
    
    def test_get_labels_batch(self):
        """Test get_labels with a batch of waveforms."""
        generator = ClusterLabelGenerator(n_clusters=10, feature_dim=16)
        
        # Manually fit
        features = np.random.randn(1000, 16).astype(np.float32)
        from sklearn.cluster import MiniBatchKMeans
        generator.kmeans = MiniBatchKMeans(n_clusters=10, random_state=42)
        generator.kmeans.fit(features)
        generator._fitted = True
        
        waveforms = np.random.randn(4, 1, 6000)  # Batch of 4
        labels = generator.get_labels(waveforms)
        
        assert isinstance(labels, torch.Tensor)
        assert labels.ndim == 2
        assert labels.shape[0] == 4
    
    def test_save_load(self, tmp_path):
        """Test saving and loading a fitted generator."""
        generator = ClusterLabelGenerator(
            n_clusters=10,
            feature_dim=16,
            feature_mode="spectrogram",  # Use spectrogram for consistent dims
        )
        
        # Extract actual features to get correct dimensions
        waveform = np.random.randn(6000)
        sample_features = generator.extract_features(waveform)
        n_features = sample_features.shape[1]
        
        # Manually fit with correct feature dimensions
        features = np.random.randn(1000, n_features).astype(np.float32)
        from sklearn.cluster import MiniBatchKMeans
        generator.kmeans = MiniBatchKMeans(n_clusters=10, random_state=42)
        generator.kmeans.fit(features)
        generator._fitted = True
        
        # Save
        save_path = tmp_path / "kmeans.pkl"
        generator.save(save_path)
        
        # Load
        loaded = ClusterLabelGenerator.load(save_path)
        
        assert loaded.n_clusters == 10
        assert loaded.feature_mode == "spectrogram"
        assert loaded._fitted
        
        # Both should be able to generate labels
        labels1 = generator.get_labels(waveform)
        labels2 = loaded.get_labels(waveform)
        
        assert labels1.shape == labels2.shape
        # Labels should be identical since same kmeans model
        assert torch.equal(labels1, labels2)
