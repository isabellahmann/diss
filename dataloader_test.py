import torch
import numpy as np
import os

from notebooks.data_loader_test import get_ensemble_all_data_loader, get_ensemble_data_loader
from test_copied_refactoring import get_data_loaders_BRATS

def compute_stats(dataloader, name, num_batches=5):
    """ Computes basic statistics for the dataset. """
    min_vals, max_vals, means, stds = [], [], [], []
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        images = batch["modality_data"]  # Shape: (B, C, H, W)
        masks = batch["selected_pred"]  # Shape: (B, H, W) or (B, C, H, W)
        
        min_vals.append(images.min().item())
        max_vals.append(images.max().item())
        means.append(images.mean().item())
        stds.append(images.std().item())
        
        print(f"{name} - Batch {i+1}:")
        print(f"  Image shape: {images.shape}, Mask shape: {masks.shape}")
        print(f"  Image range: [{images.min().item()}, {images.max().item()}]")
        print(f"  Mean: {images.mean().item()}, Std: {images.std().item()}\n")
    
    print(f"{name} Summary:")
    print(f"  Global Min: {min(min_vals)}, Global Max: {max(max_vals)}")
    print(f"  Mean of Means: {np.mean(means):.4f}, Mean of Stds: {np.mean(stds):.4f}\n")

# Load the dataloaders
# train_loader_brats, val_loader_brats = get_data_loaders_BRATS()


current_dir = os.getcwd()  # Get current working directory
flair_path = os.path.join(current_dir, "data/flair_images")
ensemble_path = os.path.join(current_dir, "data/ensemble_predictions")

train_loader_brats, val_loader_brats = get_data_loaders_BRATS(
            flair_path,
            ensemble_path,
            1,
            (64, 64))

data_dir="/home/data"

train_loader_ensemble = get_ensemble_data_loader(
    data_dir=data_dir,
    batch_size=1,
    split="train",
    modality='flair',  # Load only selected modalities
    mask_type='binary',  # Multi-class mask
    resize=(64, 64)
)

# Compute stats
compute_stats(train_loader_brats, "BRATS Dataset")
compute_stats(train_loader_ensemble, "Ensemble Dataset")
