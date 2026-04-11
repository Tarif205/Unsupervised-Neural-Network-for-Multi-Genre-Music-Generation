
import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from dataset import GrooveDataset


if __name__ == '__main__':
    mp.freeze_support()

    train_dataset = GrooveDataset("data/processed/train.npy")
    val_dataset = GrooveDataset("data/processed/val.npy")

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,  # Disable multiprocessing for now
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,  # Disable multiprocessing for now
        pin_memory=True
    )


    # TEST LOOP
    for batch in train_loader:
        print("Batch shape:", batch.shape)
        break