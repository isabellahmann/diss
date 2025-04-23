import torch
import glob
import os
import numpy as np
import nibabel as nib
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

MODALITIES = ["flair", "t1", "t2", "t1ce"]

LABEL_MAP = {
    0: 0,  # Background
    1: 1,  # NCR (Necrotic and Non-Enhancing Tumor Core)
    2: 2,  # ED (Peritumoral Edema)
    4: 3   # ET (Enhancing Tumor)
}

class BraTSDataset(Dataset):
    def __init__(self, data_dir, split="train", image_size=(64, 64), normalize=True, 
                 modalities=None, num_classes=1, return_slices=False):
        """
        Args:
            data_dir (str): Path to dataset (should contain 'train', 'val', 'test' subfolders).
            split (str): 'train', 'val', or 'test'.
            image_size (tuple): Target image size for resizing.
            normalize (bool): Whether to normalize images.
            modalities (list): Which MRI modalities to use (default: all).
            num_classes (int): Number of segmentation classes (1 for whole tumor, 4 for all tumor subtypes).
            return_slices (bool): If True, returns 2D slices (from slice 30 to -30), else returns full 3D volumes.
        """
        self.data_dir = os.path.join(data_dir)
        self.image_size = image_size
        self.normalize = normalize
        self.split = split
        self.modalities = modalities if modalities else MODALITIES
        self.num_classes = num_classes
        self.return_slices = return_slices  # If True, return 2D slices

        # Get patient folder paths
        self.patients = sorted(glob.glob(os.path.join(self.data_dir, "BraTS20_Training_*")))

        # Define augmentations (for 2D slices)
        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            # A.RandomRotate90(p=0.5),
            # A.Resize(height=240, width=240),  # Ensure consistent size after rotation
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),
            A.Normalize(mean=0, std=1) if normalize else A.NoOp(),
            ToTensorV2()
        ])

        self.val_transform = A.Compose([
            A.Normalize(mean=0, std=1) if normalize else A.NoOp(),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_path = self.patients[idx]
        
        # Load selected MRI modalities
        image_channels = []
        for modality in self.modalities:
            modality_path = os.path.join(patient_path, f"{os.path.basename(patient_path)}_{modality}.nii")
            modality_img = nib.load(modality_path).get_fdata()
            image_channels.append(modality_img)

        # Stack selected modalities into a single tensor (C, H, W, D)
        image = np.stack(image_channels, axis=0)  # Shape: (C, H, W, D)

        # Load segmentation mask
        mask_path = os.path.join(patient_path, f"{os.path.basename(patient_path)}_seg.nii")
        mask = nib.load(mask_path).get_fdata() if os.path.exists(mask_path) else None

        if mask is not None:
            mask = np.vectorize(LABEL_MAP.get)(mask)  # Convert mask labels
            if self.num_classes == 1:
                mask[mask > 0] = 1  # Convert to binary mask

        if self.return_slices:
            # Return 2D slices (from slice 30 to -30 to remove empty space)
            image = image[:, :, :, 30:-30]  # Shape: (C, H, W, D')
            mask = mask[:, :, 30:-30] if mask is not None else None  # Shape: (H, W, D')

            # Convert (C, H, W, D') to a list of (C, H, W) slices
            image_slices = [image[:, :, :, i] for i in range(image.shape[-1])]
            mask_slices = [mask[:, :, i] for i in range(mask.shape[-1])] if mask is not None else None

            # Apply augmentations for each slice (training only)
            # transform = self.train_transform if self.split == "train" else self.val_transform
            # processed_images = [transform(image=img)["image"] for img in image_slices]
            # processed_masks = [torch.tensor(mask, dtype=torch.long) for mask in mask_slices] if mask is not None else None
# Apply augmentations for each slice individually (training only)
            transform = self.train_transform if self.split == "train" else self.val_transform

            # Apply a random transformation to each slice
            processed_images = [transform(image=img)["image"] for img in image_slices]
            processed_masks = [torch.tensor(mask, dtype=torch.long) for mask in mask_slices] if mask is not None else None

            return torch.stack(processed_images), torch.stack(processed_masks) if processed_masks is not None else torch.stack(processed_images)

        else:
            # Full 3D volume (C, H, W, D)
            transform = self.train_transform if self.split == "train" else self.val_transform
            image = transform(image=image)["image"]

            if mask is not None:
                mask = torch.tensor(mask, dtype=torch.long)  # Convert to tensor

            return image, mask if mask is not None else image


def get_brats_dataloader(data_dir="/srv/thetis2/il221/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/", batch_size=4, split="train", image_size=(240, 240), 
                         normalize=True, modalities=None, num_classes=1, return_slices=False, num_workers=4):
    dataset = BraTSDataset(data_dir, split=split, image_size=image_size, normalize=normalize, 
                           modalities=modalities, num_classes=num_classes, return_slices=return_slices)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers)


# Import DataLoader
train_loader = get_brats_dataloader(
    batch_size=2,
    split="train",
    modalities=["flair"],  # Load only selected modalities
    num_classes=1,  # Multi-class mask
    return_slices=True  # Change to False for 3D volumes
)

# # Fetch a batch from the DataLoader
# for images, masks in train_loader:
#     print(f"Images Shape: {images.shape}")  # Expected (B, C, H, W) for 2D, (B, C, H, W, D) for 3D
#     print(f"Masks Shape: {masks.shape}")    # Same as images
#     break  # Stop after first batch



import matplotlib.pyplot as plt
import numpy as np

def show_image_and_mask(image, mask, num_slices=3, save_path=None):
    """
    Function to visualize images and masks, and optionally save the figure to a file.
    
    Args:
        image (torch.Tensor): The image tensor to display (shape: [C, H, W]).
        mask (torch.Tensor): The mask tensor to display (shape: [H, W]).
        num_slices (int): Number of slices to visualize. Default is 3 slices.
        save_path (str): Path to save the image. If None, the image will not be saved.
    """
    # Select a few slices (for 2D data, you can show individual channels)
    fig, axes = plt.subplots(num_slices, 2, figsize=(10, 5))
    
    # Display image and mask side-by-side for each slice
    for y in range(num_slices):
        i = y + 50
        # For 2D images, we can just display each channel (e.g., flair, t1, etc.)
        img_slice = image[i, :, :].numpy()  # Access each channel (assuming C, H, W)
        img_slice = np.squeeze(img_slice)  # Remove unnecessary singleton dimension
        
        mask_slice = mask[i, :, :].numpy()  # Extract the slice from the mask (assuming 3D mask)
        mask_slice = np.squeeze(mask_slice)  # Remove unnecessary singleton dimension

        # Display image
        axes[y, 0].imshow(img_slice, cmap='gray')
        axes[y, 0].set_title(f"Image Channel {i+1}")
        axes[y, 0].axis('off')  # Hide axes
        
        # Display mask
        axes[y, 1].imshow(mask_slice, cmap='gray')
        axes[y, 1].set_title(f"Mask Slice {i+1}")
        axes[y, 1].axis('off')  # Hide axes

    plt.tight_layout()
    
    # If save_path is provided, save the figure to that path
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')  # Save without extra margins
    
    plt.show()

import os
import matplotlib.pyplot as plt

# Get the directory where the Python file is located
script_dir = os.path.dirname(os.path.realpath(__file__))  # This will give the directory of the current script

# Specify the filename for the image
save_filename = "your_image.png"  # Specify the image file name

# Full path to save the image in the same folder as the script
save_path = os.path.join(script_dir, save_filename)

for images, masks in train_loader:
    print(f"Images Shape: {images.shape}")  # Expected shape: (B, C, H, W) or (B, C, H, W, D)
    print(f"Masks Shape: {masks.shape}")    # Expected shape: (B, H, W) or (B, H, W, D)
    
    # Assuming images are 2D slices (C, H, W)
    show_image_and_mask(images[0], masks[0], save_path=save_path)  # Show and save the images and masks

    # break  # Stop after the first batch

