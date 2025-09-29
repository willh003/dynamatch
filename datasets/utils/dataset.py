import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
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