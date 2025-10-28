from typing import Optional
import copy
import math
import pathlib

import numpy as np
import torch
import dask.array as da
from torch.utils.data import Dataset

from datasets.trajectory_utils.buffer import CompressedTrajectoryBuffer
from datasets.trajectory_utils.normalizer import LinearNormalizer, NestedDictLinearNormalizer
from datasets.trajectory_utils.obs_utils import unflatten_obs
from datasets.trajectory_utils.sampler import TrajectorySampler

class TrajectoryDatasetInterface(Dataset):
    """
    Base class for trajectory datasets.
    """
    def __len__(self) -> int:
        pass

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        pass

    def __repr__(self) -> str:
        pass

    def get_validation_dataset(self) -> "TrajectoryDataset":
        pass

def make_trajectory_dataset(
    name: str,
    buffer_dir: str,
    shape_meta: dict,
    seq_len: int,
    history_len: int = 1,
    normalize_action: bool = False,
    normalize_lowdim: bool = False,
    val_ratio: float = 0.2,
):
    """
    Factory function to create Trajectory training and validation datasets.
    
    Args:
        name: Dataset name
        buffer_dir: Path to the zarr buffer file
        shape_meta: Shape metadata for observations and actions
        seq_len: Sequence length for training
        history_len: Number of history frames to include
        normalize_action: Whether to normalize actions
        normalize_lowdim: Whether to normalize low-dimensional observations
        val_ratio: Ratio of data to use for validation
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Training dataset
    train_set = TrajectoryDataset(
        name=name,
        buffer_dir=buffer_dir,
        shape_meta=shape_meta,
        seq_len=seq_len,
        history_len=history_len,
        normalize_lowdim=normalize_lowdim,
        normalize_action=normalize_action,
        val_ratio=val_ratio,
    )

    # Validation dataset
    val_set = train_set.get_validation_dataset()
    return train_set, val_set

class TrajectoryDataset(TrajectoryDatasetInterface):
    """
    Trajectory-specific dataset that implements the same interface as TrajectoryDataset.
    
    This class handles Trajectory environments with observation space including
    cart_position, cart_velocity, pole_angle, and pole_velocity.
    """
    
    def __init__(
        self,
        name: str,
        buffer_dir: str,
        shape_meta: dict,
        seq_len: int,
        history_len: int = 1,
        normalize_lowdim: bool = False,
        normalize_action: bool = False,
        val_ratio: float = 0.0,
        num_workers: int = 8,
    ):
        """
        Initialize the TrajectoryDataset.
        
        Args:
            name: Dataset name
            buffer_dir: Path to the zarr buffer file
            shape_meta: Shape metadata for observations and actions
            seq_len: Sequence length for training
            history_len: Number of history frames to include
            normalize_action: Whether to normalize actions
            normalize_lowdim: Whether to normalize low-dimensional observations
            val_ratio: Ratio of data to use for validation
            num_workers: Number of workers for data loading
        """
        self.name = name
        self.seq_len = seq_len
        self.history_len = history_len
        self.num_workers = num_workers

        # Parse observation and action shapes
        obs_shape_meta = shape_meta["obs"]
        self._image_shapes = {}
        self._lowdim_shapes = {}
        for key, attr in obs_shape_meta.items():
            obs_type = attr["type"]
            obs_shape = tuple(attr["shape"])
            if obs_type == "rgb":
                self._image_shapes[key] = obs_shape
            elif obs_type == "low_dim":
                self._lowdim_shapes[key] = obs_shape
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")
        self._action_shape = tuple(shape_meta["action"]["shape"])

        # Compressed buffer to store episode data
        self.buffer_dir = pathlib.Path(buffer_dir).parent
        self.buffer = self._init_buffer(buffer_dir)

        # Create training-validation split
        num_episodes = self.buffer.num_episodes
        val_mask = np.zeros(num_episodes, dtype=bool)
        if val_ratio > 0:
            num_val_episodes = round(val_ratio * num_episodes)
            num_val_episodes = min(max(num_val_episodes, 1), num_episodes - 1)
            rng = np.random.default_rng(seed=0)
            val_inds = rng.choice(num_episodes, num_val_episodes, replace=False)
            val_mask[val_inds] = True
        self.train_mask = ~val_mask
        self.is_validation = False  # flag for __getitem__

        # Sampler to sample sequences from buffer
        self.sampler = TrajectorySampler(self.buffer, self.seq_len, self.train_mask)

        # Low-dim observation normalizer
        if normalize_lowdim:
            self.lowdim_normalizer = self._init_lowdim_normalizer()

        # Action normalizer
        if normalize_action:
            self.action_normalizer = self._init_action_normalizer()

    def _init_buffer(self, buffer_dir):
        """Initialize the compressed trajectory buffer."""
        # Create metadata
        metadata = {}
        
        for key, shape in self._image_shapes.items():
            metadata[f"obs.{key}"] = {"shape": shape, "dtype": np.uint8}
        for key, shape in self._lowdim_shapes.items():
            metadata[f"obs.{key}"] = {"shape": shape, "dtype": np.float32}
        metadata["action"] = {"shape": self._action_shape, "dtype": np.float32}

        # Load buffer
        buffer = CompressedTrajectoryBuffer(storage_path=buffer_dir, metadata=metadata)
        assert buffer.restored, f"Buffer not found at {buffer_dir}"
        return buffer

    def _init_lowdim_normalizer(self):
        """Initialize low-dimensional observation normalizer."""
        # Load cached normalizer statistics
        normalizer_stats_path = self.buffer_dir / "lowdim_normalizer_stats.npz"
        if normalizer_stats_path.exists():
            print(f"Loading lowdim normalizer stats from {normalizer_stats_path}")
            stats = np.load(normalizer_stats_path)
            return NestedDictLinearNormalizer(stats)

        stats = {}
        for key in self._lowdim_shapes.keys():
            data = da.from_zarr(self.buffer[f"obs.{key}"])
            min_val = data.min(axis=0).compute()
            max_val = data.max(axis=0).compute()
            scale = (max_val - min_val) / 2.0
            offset = (max_val + min_val) / 2.0
            stats[key] = (scale, offset)

        # Cache normalizer statistics
        np.savez(normalizer_stats_path, **stats)
        return NestedDictLinearNormalizer(stats)

    def _init_action_normalizer(self):
        """Initialize action normalizer."""
        # Load cached normalizer statistics
        normalizer_stats_name = f"action_normalizer_stats_len{self.seq_len}.npz"
        normalizer_stats_path = self.buffer_dir / normalizer_stats_name
        if normalizer_stats_path.exists():
            print(f"Loading action normalizer stats from {normalizer_stats_path}")
            stats = np.load(normalizer_stats_path)
            return LinearNormalizer(stats["scale"], stats["offset"])

        # Use dask to compute normalization statistics
        actions = da.from_zarr(self.buffer["action"])
        min_action = actions.min(axis=0).compute()
        max_action = actions.max(axis=0).compute()

        # Compute normalizer statistics
        scale = (max_action - min_action) / 2.0
        offset = (max_action + min_action) / 2.0

        # Cache normalizer statistics
        np.savez(normalizer_stats_path, scale=scale, offset=offset)
        return LinearNormalizer(scale, offset)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sampler)

    def __repr__(self) -> str:
        """Return string representation of the dataset."""
        return (
            "<TrajectoryDataset>\n"
            f"name: {self.name}\n"
            f"num_samples: {len(self)}\n"
            f"{self.buffer}"
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing observations and actions as torch tensors
        """
        # Sample a sequence of observations and actions from the dataset
        data = self.sampler.sample_sequence(idx)

        # Normalize low-dim observations
        if hasattr(self, "lowdim_normalizer"):
            for key in self._lowdim_shapes.keys():
                data[f"obs.{key}"] = self.lowdim_normalizer[key](data[f"obs.{key}"])

        # Normalize actions
        if hasattr(self, "action_normalizer"):
            data["action"] = self.action_normalizer(data["action"])

        # Convert data to torch tensors
        data = {k: torch.from_numpy(v) for k, v in data.items()}

        # Unflatten observations
        data = unflatten_obs(data)
        return data

    def get_validation_dataset(self):
        """
        Create a validation dataset from the current dataset.
        
        Returns:
            TrajectoryDataset: A new dataset instance configured for validation
        """
        val_set = copy.copy(self)
        val_set.train_mask = ~self.train_mask
        val_set.sampler = TrajectorySampler(self.buffer, self.seq_len, ~self.train_mask)
        val_set.is_validation = True
        return val_set