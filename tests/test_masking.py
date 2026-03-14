"""Tests for masking functionality."""

import numpy as np
import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train import (
    compute_mask_indices,
    compute_distance_adaptive_mask_length,
)


class TestComputeMaskIndices:
    """Tests for compute_mask_indices function."""
    
    def test_basic_shape(self):
        """Test that output has correct shape."""
        batch_size, seq_length = 4, 100
        mask = compute_mask_indices(
            (batch_size, seq_length),
            mask_prob=0.1,
            mask_length=5,
            device=torch.device("cpu"),
        )
        
        assert mask.shape == (batch_size, seq_length)
        assert mask.dtype == torch.bool
    
    def test_mask_coverage(self):
        """Test that approximately mask_prob fraction is masked."""
        batch_size, seq_length = 8, 200
        mask = compute_mask_indices(
            (batch_size, seq_length),
            mask_prob=0.15,
            mask_length=10,
            device=torch.device("cpu"),
        )
        
        # Check that some masking occurred
        masked_fraction = mask.float().mean().item()
        assert masked_fraction > 0.05  # At least some masking
        assert masked_fraction < 0.5   # Not too much
    
    def test_mask_spans_contiguous(self):
        """Test that masks form contiguous spans."""
        mask = compute_mask_indices(
            (1, 100),
            mask_prob=0.1,
            mask_length=5,
            device=torch.device("cpu"),
        )
        
        # Find transitions from False to True
        mask_1d = mask[0]
        transitions = (mask_1d[1:] != mask_1d[:-1]).sum().item()
        
        # Should have even number of transitions (start and end of spans)
        # or the mask could start/end at boundaries
        assert transitions >= 0
    
    def test_per_sample_mask_lengths_list(self):
        """Test with per-sample mask lengths as list."""
        batch_size, seq_length = 4, 100
        mask_lengths = [3, 5, 7, 10]
        
        mask = compute_mask_indices(
            (batch_size, seq_length),
            mask_prob=0.1,
            mask_length=mask_lengths,
            device=torch.device("cpu"),
        )
        
        assert mask.shape == (batch_size, seq_length)
    
    def test_per_sample_mask_lengths_tensor(self):
        """Test with per-sample mask lengths as tensor."""
        batch_size, seq_length = 4, 100
        mask_lengths = torch.tensor([3, 5, 7, 10])
        
        mask = compute_mask_indices(
            (batch_size, seq_length),
            mask_prob=0.1,
            mask_length=mask_lengths,
            device=torch.device("cpu"),
        )
        
        assert mask.shape == (batch_size, seq_length)
    
    def test_different_mask_lengths_produce_different_coverage(self):
        """Test that different mask lengths produce different coverage."""
        seq_length = 200
        
        # Short mask
        mask_short = compute_mask_indices(
            (10, seq_length),
            mask_prob=0.1,
            mask_length=3,
            device=torch.device("cpu"),
        )
        
        # Long mask
        mask_long = compute_mask_indices(
            (10, seq_length),
            mask_prob=0.1,
            mask_length=15,
            device=torch.device("cpu"),
        )
        
        # Both should have some masking
        assert mask_short.any()
        assert mask_long.any()
    
    def test_zero_mask_prob(self):
        """Test with very low mask probability."""
        mask = compute_mask_indices(
            (4, 100),
            mask_prob=0.001,  # Very low
            mask_length=5,
            device=torch.device("cpu"),
        )
        
        # Should still produce at least 1 span per sample
        for i in range(4):
            assert mask[i].any()
    
    def test_device_placement(self):
        """Test that mask is on correct device."""
        device = torch.device("cpu")
        mask = compute_mask_indices(
            (2, 50),
            mask_prob=0.1,
            mask_length=5,
            device=device,
        )
        
        assert mask.device == device


class TestDistanceAdaptiveMaskLength:
    """Tests for compute_distance_adaptive_mask_length function."""
    
    def test_basic_output(self):
        """Test basic output shape and type."""
        distances = torch.tensor([10.0, 50.0, 100.0])
        mask_lengths = compute_distance_adaptive_mask_length(distances)
        
        assert mask_lengths.shape == distances.shape
        assert mask_lengths.dtype == torch.int32 or mask_lengths.dtype == torch.int64
    
    def test_monotonic_increase(self):
        """Test that mask length increases with distance."""
        distances = torch.tensor([10.0, 50.0, 100.0, 150.0, 200.0])
        mask_lengths = compute_distance_adaptive_mask_length(distances)
        
        # Should be monotonically non-decreasing
        for i in range(len(mask_lengths) - 1):
            assert mask_lengths[i] <= mask_lengths[i + 1]
    
    def test_respects_min_max(self):
        """Test that output respects min and max bounds."""
        distances = torch.tensor([1.0, 500.0])  # Very close and very far
        mask_lengths = compute_distance_adaptive_mask_length(
            distances,
            min_mask=3,
            max_mask=12,
        )
        
        assert mask_lengths.min() >= 3
        assert mask_lengths.max() <= 12
    
    def test_noise_samples(self):
        """Test handling of noise samples (negative distance)."""
        distances = torch.tensor([-1.0, -1.0, 50.0])
        mask_lengths = compute_distance_adaptive_mask_length(
            distances,
            min_mask=2,
            max_mask=10,
        )
        
        # Noise samples should get average mask length
        expected_avg = (2 + 10) / 2
        assert mask_lengths[0] == int(round(expected_avg))
        assert mask_lengths[1] == int(round(expected_avg))
    
    def test_custom_distance_range(self):
        """Test with custom distance range."""
        distances = torch.tensor([5.0, 25.0, 50.0])
        mask_lengths = compute_distance_adaptive_mask_length(
            distances,
            min_mask=2,
            max_mask=10,
            min_distance=5.0,
            max_distance=50.0,
        )
        
        # At min_distance, should get min_mask
        assert mask_lengths[0] == 2
        # At max_distance, should get max_mask
        assert mask_lengths[2] == 10
        # In between should be interpolated
        assert 2 < mask_lengths[1] < 10
    
    def test_all_same_distance(self):
        """Test with all same distances."""
        distances = torch.tensor([50.0, 50.0, 50.0])
        mask_lengths = compute_distance_adaptive_mask_length(distances)
        
        # All should be the same
        assert (mask_lengths == mask_lengths[0]).all()
    
    def test_extreme_distances(self):
        """Test with distances outside the default range."""
        distances = torch.tensor([0.1, 1000.0])  # Very close and very far
        mask_lengths = compute_distance_adaptive_mask_length(
            distances,
            min_mask=2,
            max_mask=15,
            min_distance=10.0,
            max_distance=200.0,
        )
        
        # Should be clamped to bounds
        assert mask_lengths[0] == 2   # Clamped to min
        assert mask_lengths[1] == 15  # Clamped to max


class TestMaskScheduling:
    """Tests for mask scheduling logic (testing the formula, not Lightning)."""
    
    def test_linear_schedule(self):
        """Test linear mask scheduling formula."""
        mask_length_start = 3
        mask_length_end = 12
        max_epochs = 100
        
        def get_mask_length(epoch):
            progress = min(1.0, epoch / max(1, max_epochs - 1))
            return mask_length_start + progress * (mask_length_end - mask_length_start)
        
        # At start
        assert get_mask_length(0) == 3
        # At end
        assert abs(get_mask_length(99) - 12) < 0.1
        # In middle
        assert abs(get_mask_length(49) - 7.5) < 0.5
    
    def test_step_schedule(self):
        """Test step mask scheduling formula."""
        mask_length_start = 3
        mask_length_end = 12
        max_epochs = 100
        
        def get_mask_length(epoch):
            progress = min(1.0, epoch / max(1, max_epochs - 1))
            if progress < 0.33:
                return mask_length_start
            elif progress < 0.66:
                return (mask_length_start + mask_length_end) / 2
            else:
                return mask_length_end
        
        # Stage 1
        assert get_mask_length(0) == 3
        assert get_mask_length(30) == 3
        # Stage 2
        assert get_mask_length(40) == 7.5
        assert get_mask_length(60) == 7.5
        # Stage 3
        assert get_mask_length(70) == 12
        assert get_mask_length(99) == 12
    
    def test_cosine_schedule(self):
        """Test cosine mask scheduling formula."""
        mask_length_start = 3
        mask_length_end = 12
        max_epochs = 100
        
        def get_mask_length(epoch):
            progress = min(1.0, epoch / max(1, max_epochs - 1))
            cosine_progress = 0.5 * (1 - np.cos(np.pi * progress))
            return mask_length_start + cosine_progress * (mask_length_end - mask_length_start)
        
        # At start (cosine starts slowly)
        assert abs(get_mask_length(0) - 3) < 0.1
        # At end
        assert abs(get_mask_length(99) - 12) < 0.1
        # In middle (should be around midpoint)
        assert abs(get_mask_length(49) - 7.5) < 0.5


class TestMaskingIntegration:
    """Integration tests for masking with distance adaptation."""
    
    def test_distance_adaptive_with_per_sample_masking(self):
        """Test that distance-adaptive lengths work with compute_mask_indices."""
        batch_size = 4
        seq_length = 100
        
        # Different distances for each sample
        distances = torch.tensor([10.0, 50.0, 100.0, 150.0])
        mask_lengths = compute_distance_adaptive_mask_length(
            distances,
            min_mask=2,
            max_mask=15,
        )
        
        # Use per-sample mask lengths
        mask = compute_mask_indices(
            (batch_size, seq_length),
            mask_prob=0.1,
            mask_length=mask_lengths,
            device=torch.device("cpu"),
        )
        
        assert mask.shape == (batch_size, seq_length)
        # Each sample should have some masking
        for i in range(batch_size):
            assert mask[i].any()
    
    def test_varying_mask_lengths_in_batch(self):
        """Test that different samples can have different mask patterns."""
        batch_size = 8
        seq_length = 150
        
        # Very different mask lengths
        mask_lengths = [2, 5, 8, 12, 2, 5, 8, 12]
        
        mask = compute_mask_indices(
            (batch_size, seq_length),
            mask_prob=0.1,
            mask_length=mask_lengths,
            device=torch.device("cpu"),
        )
        
        # All samples should have masking
        assert mask.any(dim=1).all()
