import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage import exposure
from torchvision import transforms
from scipy.ndimage import rotate  # For arbitrary rotations
from scipy.ndimage import gaussian_filter, map_coordinates
import cv2



class BraTSDataset(Dataset):
    def __init__(self, data_dir, split='train', modality='flair', mask_type='binary', transform=None, resize=None):
        """
        PyTorch Dataset for BraTS 2020 dataset.
        
        :param data_dir: Root directory containing 'train', 'val', 'test' subdirectories
        :param split: 'train', 'val', or 'test'
        :param modality: 'flair', 't1', 't2', 't1ce', etc.
        :param mask_type: 'binary' (tumor vs no tumor) or 'multiclass' (separate tumor classes)
        :param transform: Torchvision transforms for augmentation
        :param resize (tuple, optional): New (height, width) to resize images & masks.
        """
        self.data_dir = os.path.join(data_dir, split, modality)
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
        print(patient)
        patient_path = os.path.join(self.data_dir, patient)
        
        # Load modality (e.g., flair, t1, t2, etc.)
        modality_data = np.load(os.path.join(patient_path)).astype(np.float32)
        
        # Load corresponding segmentation mask
        seg_data = np.load(os.path.join(patient_path.replace(self.modality, 'seg'))).astype(np.float32)

                # Process mask based on selected mask type
        if self.mask_type == 'binary':
            seg_data = (seg_data > 0).astype(np.float32)  # Convert to 0 (background) and 1 (tumor)
        elif self.mask_type == 'multiclass':
            seg_data = seg_data.astype(np.int32)  # Keep original multi-class mask
        else:
            raise ValueError(f"Invalid mask_type '{self.mask_type}'. Choose 'binary' or 'multiclass'.")
        
                # Resize if needed
        if self.resize:
            new_h, new_w = self.resize
            modality_data = cv2.resize(modality_data, (new_w, new_h), interpolation=cv2.INTER_LINEAR)  # Bilinear for images
            seg_data = cv2.resize(seg_data, (new_w, new_h), interpolation=cv2.INTER_NEAREST)  # Nearest for masks (preserves labels)
        
        # Apply on-the-fly transformations
        if self.transform:
            modality_data, seg_data = self.transform(modality_data, seg_data)
        
        # Make sure the arrays are contiguous before converting to tensors
        modality_data = torch.tensor(modality_data.copy(), dtype=torch.float32)
        seg_data = torch.tensor(seg_data.copy(), dtype=torch.float32)
        
        return modality_data, seg_data, patient
    

class BraTSEnsembleDataset(Dataset):
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
        self.ensemble_pred_dir = os.path.join(data_dir, split, 'ensemble_pred')  # Directory for ensemble predictions
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
        
        # Randomly select one of the ensemble predictions
        # selected_pred = random.choice(ensemble_preds)  # Choose a random prediction


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

        # Apply on-the-fly transformations if any
        # if self.transform:
        #     modality_data, selected_pred = self.transform(modality_data, selected_pred)
        

        # Make sure the arrays are contiguous before converting to tensors
        modality_data = torch.tensor(modality_data.copy(), dtype=torch.float32)
        ensemble_preds = torch.tensor(ensemble_preds.copy(), dtype=torch.float32)
        
        return {"modality_data": modality_data, "selected_pred": ensemble_preds, "patient": patient}
    

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
        self.ensemble_pred_dir = os.path.join(data_dir, split, 'ensemble_pred')  # Directory for ensemble predictions
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
        # seg_data = np.load(os.path.join(patient_path.replace(self.modality, 'seg'))).astype(np.float32)

                # Randomly select one of the ensemble predictions
        # selected_pred = random.choice(ensemble_preds)  # Choose a random prediction
        # selected_pred = np.random.choice(ensemble_preds)
        random_idx = np.random.randint(0, ensemble_preds.shape[0])  # Pick a random prediction index
        selected_pred = ensemble_preds[random_idx]  # Shape: (height, width)


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
        return {"modality_data": modality_data, "selected_pred": selected_pred, "patient": patient}





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
    
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
import random

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
        # Normalize to [0, 1] and apply a standard mean and std (based on the modality)
        img = img / img.max()  # Normalize to 0-1 range
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
    

def get_data_loader(data_dir="/srv/thetis2/il221/BraTS2020_Processed", modality='flair', mask_type='binary', batch_size=8, num_workers=4, split='train', resize=(128, 128)):
    """
    Returns a DataLoader for the specified split (train, val, or test).
    
    Parameters:
        data_dir (str): Path to the dataset.
        modality (str): MRI modality (default: 'flair').
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of worker threads for data loading.
        split (str): Dataset split ('train', 'val', 'test').
        resize (tuple): Target size for resizing images.
    
    Returns:
        DataLoader: The requested DataLoader (train, val, or test).
    """

    augmentations = Compose([
        RandomVerticalFlip(),
        RandomRotation(),
        RandomGammaCorrection(gamma_range=(0.5, 2.0)),
        RandomGaussianNoise(),
        ElasticDeformation()
    ])
    
    normalize = Normalize(mean=0.5, std=0.5)
    
    # Apply augmentations only for training
    transform = Compose([augmentations, normalize]) if split == 't' else Normalize(mean=0.5, std=0.5)

    # Create dataset
    dataset = BraTSDataset(data_dir, split=split, modality=modality, mask_type=mask_type, transform=transform, resize=resize)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'), drop_last=(split == 'train'), num_workers=num_workers)

    return dataloader


def get_ensemble_data_loader(data_dir="/srv/thetis2/il221/BraTS2020_Processed", modality='flair', mask_type='binary', batch_size=8, num_workers=4, split='train', resize=(128, 128)):
    """
    Returns a DataLoader for the specified split (train, val, or test).
    
    Parameters:
        data_dir (str): Path to the dataset.
        modality (str): MRI modality (default: 'flair').
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of worker threads for data loading.
        split (str): Dataset split ('train', 'val', 'test').
        resize (tuple): Target size for resizing images.
    
    Returns:
        DataLoader: The requested DataLoader (train, val, or test).
    """

    # Define augmentations (only for training)
    augmentations = Compose([
        RandomVerticalFlip(),
        RandomRotation(),
        RandomGammaCorrection(gamma_range=(0.5, 2.0)),
        RandomGaussianNoise(),
        ElasticDeformation()
    ])
    
    normalize = Normalize(mean=0.5, std=0.5)
    
    # Apply augmentations only for training
    # transform = Compose([augmentations, normalize]) if split == 'train' else Normalize(mean=0.5, std=0.5)
    transform =  Normalize(mean=0.5, std=0.5)

    # Create dataset
    dataset = BraTSEnsembleDataset(data_dir, split=split, modality=modality, mask_type=mask_type, transform=transform, resize=resize)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'), drop_last=(split == 'train'), num_workers=num_workers)

    return dataloader


def get_ensemble_all_data_loader(data_dir="/srv/thetis2/il221/BraTS2020_Processed", modality='flair', mask_type='binary', batch_size=8, num_workers=4, split='train', resize=(128, 128)):
    # Define augmentations (only for training)
    augmentations = Compose([
        RandomVerticalFlip(),
        RandomRotation(),
        RandomGammaCorrection(gamma_range=(0.5, 2.0)),
        RandomGaussianNoise(),
        ElasticDeformation()
    ])
    
    normalize = Normalize(mean=0.5, std=0.5)
    
    # Apply augmentations only for training
    transform = Compose([augmentations, normalize]) if split == 'train' else Normalize(mean=0.5, std=0.5)

    # Create dataset
    dataset = BraTSEnsembleAllDataset(data_dir, split=split, modality=modality, mask_type=mask_type, transform=None, resize=resize)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'), drop_last=(split == 'train'), num_workers=num_workers)

    return dataloader


# # Testing the data loader and augmentations
# if __name__ == "__main__":
#     data_dir = "/srv/thetis2/il221/BraTS2020_Processed"  # Change this to your dataset location
#     modality = 'flair'  # Change this to any other modality ('t1', 't2', etc.)
    
#     train_loader = get_ensemble_all_data_loader(
#         batch_size=8,
#         split="train",
#         modality='flair',  # Load only selected modalities
#         mask_type='binary',  # Multi-class mask
#         resize=(64, 64)
#     )

#     val_loader = get_ensemble_all_data_loader(
#         batch_size=8,
#         split="val",
#         modality='flair',  # Load only selected modalities
#         mask_type='binary',  # Multi-class mask
#         resize=(64, 64)
#     )


#     # Check one batch
# import os
# import matplotlib.pyplot as plt


# # Loop over the data loader to get batches
# for batch_idx, batch in enumerate(train_loader):
#     modality_data, gt, seg_data, patient = batch
#     print(f"Modality data shape: {modality_data.shape}, gt data shape: {gt.shape} Segmentation data shape: {seg_data.shape}")

#     # Create a figure with subplots to display 8 images and their corresponding masks
#     fig, axes = plt.subplots(3, figsize=(10, 20))  # 8 rows and 2 columns: one for image, one for mask

#     # Loop through each of the 8 samples in the batch
#     for i in range(modality_data.shape[0]):
#         # Get the image and mask data for the current sample
#         first_image = modality_data[i].numpy()
#         first_mask = seg_data[i].numpy()
#         firt_gt = gt[i].numpy()
#         print(f"Modality data shape: {first_image.min(), first_image.max()}, gt data shape: {first_mask.min(), first_mask.max()} Segmentation data shape: {firt_gt.min(), firt_gt.max()}")


#         # Display the image in the first column
#         axes[0].imshow(first_image, cmap='gray')
#         axes[0].set_title(f"Image {i+1}")
#         axes[0].axis('off')

#         # Display the ground truth mask in the second column
#         axes[1].imshow(firt_gt, cmap='gray')  # Use transparency for better visibility
#         axes[1].set_title(f"GT Mask")
#         axes[1].axis('off')


#         axes[2].imshow(first_mask, cmap='gray')
#         axes[2].set_title(f"Pred")
#         axes[2].axis('off')

#     # Define the filename for saving the figure
#     save_path = f"batch_{batch_idx + 1}_images_masks.png"
    
#     # Save the figure
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()

#     print(f"Saved batch {batch_idx + 1} images and masks as {save_path}")
#     break  


# import matplotlib.pyplot as plt
# import torch

# import matplotlib.pyplot as plt
# import numpy as np

# # Loop over the data loader to get batches
# for batch_idx, batch in enumerate(train_loader):
#     modality_data, gt, seg_data, patient = batch
#     print(f"Modality data shape: {modality_data.shape}, GT shape: {gt.shape}, Segmentation shape: {seg_data.shape}")

#     num_preds = seg_data.shape[1]  # Number of predicted masks per image

#     # Create a figure with subplots (8 rows, 2 + num_preds columns)
#     fig, axes = plt.subplots(8, num_preds + 2, figsize=(15, 20))  # Extra column for GT

#     # Loop through each of the 8 samples in the batch
#     for i in range(min(8, modality_data.shape[0])):  # Ensure we don't exceed batch size
#         # Get the image and mask data for the current sample
#         first_image = modality_data[i].numpy()
#         first_gt = gt[i].numpy()  # Ground truth mask

#         # Display the image in the first column
#         axes[i, 0].imshow(first_image, cmap='gray')
#         axes[i, 0].set_title(f"Image {i+1}")
#         axes[i, 0].axis('off')

#         # Display the ground truth mask in the second column
#         axes[i, 1].imshow(first_gt, cmap='jet', alpha=0.5)  # Use transparency for better visibility
#         axes[i, 1].set_title(f"GT Mask {i+1}")
#         axes[i, 1].axis('off')

#         # Display all predictions in subsequent columns
#         for j in range(num_preds):
#             first_mask = seg_data[i, j].numpy()  # Predicted mask

#             axes[i, j + 2].imshow(first_mask, cmap='jet', alpha=0.5)
#             axes[i, j + 2].set_title(f"Pred {j+1}")
#             axes[i, j + 2].axis('off')

#     # Define the filename for saving the figure
#     save_path = f"batch_{batch_idx + 1}_images_gt_preds.png"

#     # Save the figure
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()

#     print(f"Saved batch {batch_idx + 1} images, GT masks, and predictions as {save_path}")
#     break  # Exit after saving one batch (remove this break if you want to save multiple batches)



# path = '/srv/thetis2/il221/BraTS2020_Processed/train/ensemble_pred/BraTS20_Training_241_slice_80.npy'
# gt = '/srv/thetis2/il221/BraTS2020_Processed/train/seg/BraTS20_Training_241_slice_80.npy'

# ensemble_preds = np.load(path).astype(np.float32)
# gt_pred = np.load(gt).astype(np.float32)

# # Create a figure with subplots (8 rows, 2 + num_preds columns)
# fig, axes = plt.subplots(1, 9, figsize=(15, 20))  # Extra column for GT

# # Loop through each of the 8 samples in the batch
# for i in range(8):  # Ensure we don't exceed batch size
#     # Get the image and mask data for the current sample
#     first_gt = gt_pred  # Ground truth mask

#     # Display the image in the first column

#     # Display the ground truth mask in the second column
#     axes[0].imshow(first_gt, cmap='jet', alpha=0.5)  # Use transparency for better visibility
#     axes[0].set_title(f"GT Mask")
#     axes[0].axis('off')

#     # Display all predictions in subsequent columns
#     for j in range(num_preds):
#         first_mask = seg_data[i, j].numpy()  # Predicted mask

#         axes[j + 1].imshow(first_mask, cmap='jet', alpha=0.5)
#         axes[j + 1].set_title(f"Pred {j+1}")
#         axes[j + 1].axis('off')

# # Define the filename for saving the figure
# save_path = f"batch_{batch_idx + 1}_images_gt_preds_try_again.png"

# # Save the figure
# plt.tight_layout()
# plt.savefig(save_path)
# plt.close()

# checking cuda
# print(f"Using CUDA Device Index: {torch.cuda.current_device()}")  # Should print 0
# print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")