import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
import cv2
from scipy.ndimage import gaussian_filter

from torchvision.transforms import functional as TF

import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import montage
from skimage.transform import resize
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class BraTSDataset2D(Dataset):
    def __init__(self, folders, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the dataset directory.
            transform (callable, optional): Optional transform to apply on the data.
        """
        self.root_dir = root_dir
        self.patients = folders
        self.transform = transform

        # Precompute all slices for indexing
        self.slice_indices = []
        for patient_idx, patient_folder in enumerate(self.patients):
            patient_path = os.path.join(root_dir, patient_folder)
            flair = nib.load(os.path.join(patient_path, patient_folder + '_flair.nii')).get_fdata()
            num_slices = flair.shape[2]
            # Only include slices from 50:-50
            valid_slices = range(50, num_slices - 50)
            for slice_idx in valid_slices:
                self.slice_indices.append((patient_idx, slice_idx))

        print(len(self.slice_indices))

    def __len__(self):
        # Number of valid slices across all patients
        return len(self.slice_indices)

    def __getitem__(self, idx):
        # Get patient index and slice index
        print('idx', idx)
        patient_idx, slice_idx = self.slice_indices[idx]

        # Get patient folder
        patient_folder = os.path.join(self.root_dir, self.patients[patient_idx])

        # Load modalities and segmentation mask
        flair = nib.load(os.path.join(patient_folder, self.patients[patient_idx] + '_flair.nii')).get_fdata()
        # t1 = nib.load(os.path.join(patient_folder, self.patients[patient_idx] + '_t1.nii')).get_fdata()
        t1ce = nib.load(os.path.join(patient_folder, self.patients[patient_idx] + '_t1ce.nii')).get_fdata()
        # t2 = nib.load(os.path.join(patient_folder, self.patients[patient_idx] + '_t2.nii')).get_fdata()
        seg = nib.load(os.path.join(patient_folder, self.patients[patient_idx] + '_seg.nii')).get_fdata()

        flair_slice = flair[:, :, slice_idx]
        # t1_slice = t1[:, :, slice_idx]
        t1ce_slice = t1ce[:, :, slice_idx]
        # t2_slice = t2[:, :, slice_idx]
        seg_slice = seg[:, :, slice_idx]

        modalities = np.stack([flair_slice, t1ce_slice], axis=0)

        # Apply transforms
        if self.transform:
            modalities, seg_slice = self.transform(modalities, seg_slice)

        return torch.tensor(modalities, dtype=torch.float32), torch.tensor(seg_slice, dtype=torch.long)

def default_transform(modalities, seg):
    modalities = (modalities - modalities.mean(axis=(1, 2), keepdims=True)) / modalities.std(axis=(1, 2), keepdims=True)

    modalities = torch.tensor(modalities, dtype=torch.float32)
    seg[seg != 0] = 1
    seg = torch.tensor(seg, dtype=torch.long)

    modalities = TF.resize(modalities, [64, 64])
    seg = TF.resize(seg.unsqueeze(0), [64, 64], interpolation=TF.InterpolationMode.NEAREST).squeeze(0)

    return modalities, seg

class SyntheticMRIDataset(Dataset):
    def __init__(self, num_samples=50, img_size=64, tumor_probability=0.7, num_masks=3):
        self.num_samples = num_samples
        self.img_size = img_size
        self.tumor_probability = tumor_probability
        self.num_masks = num_masks

        self.images, self.masks = self.generate_toy_dataset(num_samples, img_size, tumor_probability)

    def generate_toy_dataset(self, num_samples, img_size, tumor_probability):
        """Generate synthetic brain MRI-like images with optional tumors."""
        X = []
        Y = []

        for _ in range(num_samples):
            brain_texture = np.random.normal(loc=0.5, scale=0.2, size=(img_size, img_size))
            brain_texture = gaussian_filter(brain_texture, sigma=img_size / 10)  # Smooth it

            tumor_mask = np.zeros((img_size, img_size), dtype=np.float32)

            if np.random.rand() < tumor_probability:  # 70% chance of a tumor
                center = (np.random.randint(8, img_size-8), np.random.randint(8, img_size-8))  # Keep tumor within bounds
                axes = (np.random.randint(3, 8), np.random.randint(3, 8))  # Tumor size
                angle = np.random.randint(0, 180)  # Tumor orientation
                cv2.ellipse(tumor_mask, center, axes, angle, 0, 360, (1,), -1)

            brain_texture = (brain_texture - brain_texture.min()) / (brain_texture.max() - brain_texture.min())

            synthetic_mri = brain_texture + tumor_mask * 0.5
            synthetic_mri = np.clip(synthetic_mri, 0, 1)

            tumor_mask = tumor_mask.reshape(1, img_size, img_size)

            X.append(torch.tensor(synthetic_mri, dtype=torch.float32).unsqueeze(0))
            Y.append(torch.tensor(tumor_mask, dtype=torch.float32))

        return torch.stack(X), torch.stack(Y)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]