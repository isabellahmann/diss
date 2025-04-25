import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from notebooks.data_loader import get_ensemble_all_data_loader

def compute_stats(dataloader, name, num_batches=5, output_dir="stats_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    min_vals, max_vals, means, stds = [], [], [], []
    mask_means, mask_stds, mask_nonzeros = [], [], []
    pixel_values = []
    mask_values = []

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        images = batch["modality_data"]  # Shape: (B, C, H, W)
        masks = batch["selected_pred"]  # Shape: (B, C, H, W) or (B, H, W)
        images = images.float()
        masks = masks.float()

        # Flatten for histograms
        pixel_values.append(images.view(-1).cpu().numpy())
        mask_values.append(masks.view(-1).cpu().numpy())

        min_vals.append(images.min().item())
        max_vals.append(images.max().item())
        means.append(images.mean().item())
        stds.append(images.std().item())

        mask_means.append(masks.mean().item())
        mask_stds.append(masks.std().item())
        mask_nonzeros.append(torch.count_nonzero(masks).item())

        print(f"{name} - Batch {i+1}:")
        print(f"  Image shape: {images.shape}, Mask shape: {masks.shape}")
        print(f"  Image range: [{images.min().item()}, {images.max().item()}]")
        print(f"  Image mean: {images.mean().item():.4f}, std: {images.std().item():.4f}")
        print(f"  Mask mean: {masks.mean().item():.4f}, std: {masks.std().item():.4f}, non-zero: {torch.count_nonzero(masks).item()}\n")

    print(f"{name} Summary:")
    print(f"  Image Min: {min(min_vals):.4f}, Max: {max(max_vals):.4f}")
    print(f"  Mean of Means: {np.mean(means):.4f}, Mean of Stds: {np.mean(stds):.4f}")
    print(f"  Mask Mean: {np.mean(mask_means):.4f}, Mask Std: {np.mean(mask_stds):.4f}, Avg Non-zero: {np.mean(mask_nonzeros):.2f}")

    # Plot and save histograms
    pixel_values = np.concatenate(pixel_values)
    mask_values = np.concatenate(mask_values)

    plt.figure(figsize=(8, 4))
    plt.hist(pixel_values, bins=50, color='blue', alpha=0.7)
    plt.title(f"{name} Image Pixel Intensity Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_image_hist.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(mask_values, bins=50, color='green', alpha=0.7)
    plt.title(f"{name} Mask Value Histogram")
    plt.xlabel("Mask Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_mask_hist.png"))
    plt.close()

    print(f"Histograms saved to: {output_dir}\n")

current_dir = os.getcwd()
flair_path = os.path.join(current_dir, "data/flair_images")
ensemble_path = os.path.join(current_dir, "data/ensemble_predictions")

# Load BRATS loaders
train_loader_brats, val_loader_brats = get_ensemble_all_data_loader(
    flair_path,
    ensemble_path,
    batch_size=1,
    image_size=(64, 64)
)

# Load Ensemble loaders
data_dir = "/home/data"
train_loader_ensemble = get_ensemble_all_data_loader(
    data_dir=data_dir,
    batch_size=1,
    split="train",
    modality='flair',
    mask_type='binary',
    resize=(64, 64)
)

val_loader_ensemble = get_ensemble_all_data_loader(
    data_dir=data_dir,
    batch_size=1,
    split="val",
    modality='flair',
    mask_type='binary',
    resize=(64, 64)
)

compute_stats(train_loader_brats, "BRATS_Train", num_batches=10)
compute_stats(val_loader_brats, "BRATS_Val", num_batches=10)

compute_stats(train_loader_ensemble, "Ensemble_Train", num_batches=10)
compute_stats(val_loader_ensemble, "Ensemble_Val", num_batches=10)
