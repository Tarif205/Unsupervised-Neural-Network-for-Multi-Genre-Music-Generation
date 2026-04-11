import numpy as np
import torch
from torch.utils.data import Dataset

class GrooveDataset(Dataset):
    def __init__(self, npy_path, seq_len=500, feature_size=88):
        self.data = np.load(npy_path)

        self.seq_len = seq_len
        self.feature_size = feature_size

        # ensure reshape is possible
        assert self.data.shape[1] == seq_len * feature_size, \
            f"Expected {seq_len * feature_size}, got {self.data.shape[1]}"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]

        x = torch.tensor(x, dtype=torch.float32)

        # 🔥 convert 1D → 3D sequence
        x = x.view(self.seq_len, self.feature_size)

        return x