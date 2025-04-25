import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt
from skimage import exposure
from torchvision import transforms
from scipy.ndimage import rotate  # For arbitrary rotations
from scipy.ndimage import gaussian_filter, map_coordinates
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
import random

class BraTSEnsembleAllDataset(Dataset):
    def __init__(self, data_dir, split='train', modality='flair', mask_type='binary', transform=None, resize=None):
        """
        PyTorch Dataset for BraTS 2020 dataset.
        
        :param data_dir: Root directory containing 'train', 'val', 'test' subdirectories
        :param split: 'train', 'val', or 'test'
        :param modality: 'flair', 't1', 't2', 't1ce', etc.
        :param mask_type: 'binary' (tumor vs no tumor) or 'multiclass' (separate tumor classes)
        :param transform: Torchvision transforms for augmentation
        """
        self.data_dir = os.path.join(data_dir, split, modality)
        self.ensemble_pred_dir = os.path.join(data_dir, split, 'ensemble_pred_logit')  # Directory for ensemble predictions
        self.modality = modality
        self.mask_type = mask_type
        self.transform = transform
        self.resize = resize
        self.patients = sorted(os.listdir(self.data_dir))
        
        # Ensure that the split folder exists
        if not self.patients:
            raise ValueError(f"No data found in {self.data_dir}")

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient = self.patients[idx]
        patient_path = os.path.join(self.data_dir, patient)
        ensemble_pred_path = os.path.join(self.ensemble_pred_dir, patient)  # Path to the ensemble predictions
        
        # Load modality (e.g., flair, t1, t2, etc.)
        modality_data = np.load(os.path.join(patient_path)).astype(np.float32)
        
        # Load corresponding ensemble prediction (from ensemble_pred directory)
        ensemble_preds = np.load(os.path.join(ensemble_pred_path)).astype(np.float32)

        #gt
        gt_path = os.path.join(patient_path.replace(self.modality, 'seg'))
        gt_mask = np.load(gt_path).astype(np.float32)
        if self.resize:
            new_h, new_w = self.resize
            gt_mask = cv2.resize(gt_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        gt_mask = torch.tensor(gt_mask.copy(), dtype=torch.float32)

        # Randomly select one of the ensemble predictions
        # selected_pred = random.choice(ensemble_preds)  # Choose a random prediction
        # selected_pred = np.random.choice(ensemble_preds)
        # random_idx = np.random.randint(0, ensemble_preds.shape[0])  # Pick a random prediction index
        # selected_pred = ensemble_preds[random_idx]  # Shape: (height, width)
        selected_pred = ensemble_preds  # Shape: [8, H, W]



        # selected_pred = (selected_pred - selected_pred.min()) / (selected_pred.max() - selected_pred.min())

        # print(f"Unique values in selected_pred for {patient}: {np.unique(selected_pred)}")


        # # Process mask based on selected mask type (if applicable)
        # if self.mask_type == 'binary':
        #     selected_pred = (selected_pred > 0).astype(np.float32)  # Convert to 0 (background) and 1 (tumor)
        # elif self.mask_type == 'multiclass':
        #     selected_pred = selected_pred.astype(np.int32)  # Keep original multi-class mask
        # else:
        #     raise ValueError(f"Invalid mask_type '{self.mask_type}'. Choose 'binary' or 'multiclass'.")
        

        if self.resize:
            new_h, new_w = self.resize
            modality_data = cv2.resize(modality_data, (new_w, new_h), interpolation=cv2.INTER_LINEAR)  # Bilinear for images
            # seg_data = cv2.resize(seg_data, (new_w, new_h), interpolation=cv2.INTER_NEAREST)  # Nearest for masks (preserves labels)
            selected_pred = cv2.resize(selected_pred, (new_w, new_h), interpolation=cv2.INTER_NEAREST)  # Nearest for masks (preserves labels)

        # Apply on-the-fly transformations if any
        if self.transform:
            # modality_data, selected_pred, seg_data = self.transform(modality_data, selected_pred, seg_data)
            modality_data, selected_pred = self.transform(modality_data, selected_pred)
            # for pred in ensemble_preds:
            #     pred = self.transform(pred)
        
        # Make sure the arrays are contiguous before converting to tensors
        modality_data = torch.tensor(modality_data.copy(), dtype=torch.float32)
        selected_pred = torch.tensor(selected_pred.copy(), dtype=torch.float32)
        # seg_data = torch.tensor(seg_data.copy(), dtype=torch.float32)

        # return modality_data, selected_pred, patient
        return {
            "modality_data": modality_data,
            "selected_pred": selected_pred,  # shape [N, H, W]
            "gt_mask": gt_mask,
            "patient": patient
        }



# Custom augmentation class for applying the same transformations to both image and mask
class RandomVerticalFlip:
    def __call__(self, img, mask):
        if random.random() > 0.5:
            img = np.flip(img, axis=0)
            # pred = np.flip(pred, axis=0)
            mask = np.flip(mask, axis=0)
        return img, mask


class RandomRotation:
    def __init__(self, angle_range=(-15, 15)):
        self.angle_range = angle_range

    def __call__(self, img, mask):
        # Randomly choose an angle between the range
        angle = random.uniform(*self.angle_range)
        
        # Rotate the image and mask by the selected angle
        # Mode 'nearest' avoids boundary artifacts by filling outside pixels with the nearest value
        img_rotated = rotate(img, angle, reshape=False, mode='nearest', order=1)  # 1 for bilinear interpolation
        mask_rotated = rotate(mask, angle, reshape=False, mode='nearest', order=0)  # 0 for nearest-neighbor (for segmentation)
        # pred_rotated = rotate(pred, angle, reshape=False, mode='nearest', order=0)  # 0 for nearest-neighbor (for segmentation)

        return img_rotated, mask_rotated


class RandomGammaCorrection:
    def __init__(self, gamma_range=(0.5, 2.0)):
        self.gamma_range = gamma_range

    def __call__(self, img, mask):
        gamma = random.uniform(*self.gamma_range)
        img = exposure.adjust_gamma(img, gamma)  # Apply gamma correction using skimage's adjust_gamma
        return img, mask
    
class RandomGaussianNoise:
    def __init__(self, mean=0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        noise = np.random.normal(self.mean, self.std, img.shape)
        img = np.clip(img + noise, 0, 1)  # Ensure image remains within [0, 1] range
        return img, mask
    

class ElasticDeformation:
    def __init__(self, alpha=1.0, sigma=20):
        """
        Elastic deformation for augmenting images.
        
        :param alpha: Magnitude of the displacement field. Higher values lead to more severe deformations.
        :param sigma: Standard deviation for the Gaussian kernel used to smooth the displacement field.
        """
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, img, mask):
        """
        Apply elastic deformation to both the image and its corresponding mask.
        
        :param img: The image to deform (2D).
        :param mask: The corresponding mask to deform (2D).
        :return: Deformed image and mask.
        """
        random_state = np.random.RandomState(None)

        # Generate random displacement fields
        shape = img.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha

        # Create a meshgrid of coordinates
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

        # Apply displacement
        distorted_img = map_coordinates(img, [y + dy, x + dx], order=1, mode='nearest')
        distorted_mask = map_coordinates(mask, [y + dy, x + dx], order=1, mode='nearest')
        # distorted_pred = map_coordinates(pred, [y + dy, x + dx], order=1, mode='nearest')

        return distorted_img, distorted_mask


# Define normalization transform
class Normalize:
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        img_max = img.max()
        if img_max > 0:
            img = img / img_max
        else:
            img = img
        img = (img - self.mean) / self.std
        return img, mask


# Create a pipeline of transformations
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask
    


def get_ensemble_all_data_loader(data_dir="/srv/thetis2/il221/BraTS2020_Processed", modality='flair', mask_type='binary', batch_size=8, num_workers=4, split='train', data_size=None, resize=(128, 128)):

    # Apply normalization
    normalize = Normalize(mean=0.5, std=0.5)
    transform = Compose([normalize])

    # Create dataset
    dataset = BraTSEnsembleAllDataset(data_dir, split=split, modality=modality, mask_type=mask_type, transform=transform, resize=resize)

    # If data_size is specified, randomly sample from the dataset
    data_size = len(dataset) if not data_size else data_size

    # Generate list of indices
    train_indices = list(range(len(dataset)))

    # Create samplers for each split
    sampler = SubsetRandomSampler(train_indices[:data_size])

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, sampler=sampler, pin_memory=True)

    return dataloader
