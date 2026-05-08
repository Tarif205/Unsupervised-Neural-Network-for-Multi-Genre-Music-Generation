# src/train.py
import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from dataset import GrooveDataset
from config import BATCH_SIZE, DEVICE


def get_dataloaders():
    train_dataset = GrooveDataset("train")
    val_dataset   = GrooveDataset("val")
    test_dataset  = GrooveDataset("test")

    train_loader = DataLoader(
        train_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = 0,
        pin_memory  = (DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = (DEVICE.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = (DEVICE.type == "cuda"),
    )
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    mp.freeze_support()

    print(f"Device : {DEVICE}")

    train_loader, val_loader, test_loader = get_dataloaders()

    # Sanity check
    for batch in train_loader:
        print(f"Batch shape : {batch.shape}")           # expect (32, 500, 88)
        print(f"Value range : [{batch.min():.3f}, {batch.max():.3f}]")  # expect [0.0, 1.0]
        break

    print("DataLoader OK.")