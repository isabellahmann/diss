# Author: Ivor Simpson, University of Sussex (i.simpson@sussex.ac.uk)
# Purpose: Blocks of networks

from models.model import SUPNEncoderDecoder
from supn_base.supn_data import SUPNData
from supn_base.supn_distribution import SUPN

import torch.nn as nn
from enum import Enum

import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import sys
import os

# Add the 'models' directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models')))

# Now import model
from models.model import SUPNEncoderDecoder


class BRATSDataset(Dataset):
    def __init__(self, flair_dir='dataset/2afc', masks_dir='dataset/2afc', dataset_type="train", img_size=(256, 256), num_masks=5, split_ratio=0.8):
        """
        FLAIR dataset with corresponding masks from an ensemble.
        The dataset will load both FLAIR images and 5 masks corresponding to each FLAIR image.

        :param flair_dir: Root directory containing FLAIR images.
        :param masks_dir: Root directory containing mask images.
        :param dataset_type: 'train' or 'val' to specify which split to use.
        :param img_size: Target size of the images.
        :param num_masks: Number of masks to load per FLAIR image.
        :param split_ratio: Ratio to split dataset into train and validation (e.g., 0.8 means 80% train, 20% validation).
        """
        super(BRATSDataset, self).__init__()
        self.flair_dir = flair_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.num_masks = num_masks
        self.split_ratio = split_ratio

        # Get list of all FLAIR images
        self.flair_filenames = self._get_image_filenames(self.flair_dir)

        # Split into train and validation based on split_ratio
        num_train = int(len(self.flair_filenames) * self.split_ratio)
        random.shuffle(self.flair_filenames)

        # Assign images to train/val based on the split ratio
        if dataset_type == "train":
            self.flair_filenames = self.flair_filenames[:num_train]
        elif dataset_type == "val":
            self.flair_filenames = self.flair_filenames[num_train:]
        else:
            raise ValueError("dataset_type should be either 'train' or 'val'.")

    def _get_image_filenames(self, directory):
        """
        Get all image filenames (excluding directories) from the given folder.
        """
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def _get_mask_filenames(self, flair_filename):
        """
        For each FLAIR image, get corresponding mask filenames.
        Assumes the masks are named based on the FLAIR image with a mask index (e.g., flair_image_1_mask_1.png).
        """
        base_name = os.path.splitext(os.path.basename(flair_filename))[0]
        mask_filenames = []
        # for i in range(1, self.num_masks + 1):
        for i in range(1, 5):
            mask_filename = os.path.join(self.masks_dir, f"{base_name}_ensemble_{i}.png")
            if not mask_filename.endswith("_ensemble_2.png"):  # Exclude _ensemble_2
                mask_filenames.append(mask_filename)
        return mask_filenames

        # FileNotFoundError: [Errno 2] No such file or directory: '/content/drive/MyDrive/Diss/ensemble_predictions/patient_80_9_mask_1.png'


    def __len__(self):
        return len(self.flair_filenames)

    def __getitem__(self, index):
        """
        Load FLAIR image and corresponding masks.
        """
        # Load FLAIR image
        flair_image_path = self.flair_filenames[index]
        flair_image = Image.open(flair_image_path).convert("L")

        # Load masks corresponding to this FLAIR image
        mask_filenames = self._get_mask_filenames(flair_image_path)
        masks = [Image.open(mask_filename).convert("L") for mask_filename in mask_filenames]

        # Resize FLAIR image and masks to target size
        # flair_image = flair_image.resize(self.img_size, Image.BICUBIC)
        # masks = [mask.resize(self.img_size, Image.NEAREST) for mask in masks]

        # # Convert images to tensors
        flair_image = transforms.ToTensor()(flair_image)
        masks = [transforms.ToTensor()(mask) for mask in masks]

        # Apply normalization to the FLAIR image
        # normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        # flair_image = normalize(flair_image)

        # Return FLAIR image and masks as a dictionary
        return {
            "flair_image": flair_image,
            "masks": torch.stack(masks)  # Stack the masks along the first dimension
        }

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

def get_data_loaders_BRATS(flair_dir='dataset/2afc', masks_dir = '', batchsize=64, img_size=(64, 64), train_size=None, val_size=None, num_scales=4, num_levels=5, colour = False):
    # Initialize train and validation datasets
    train_dataset = BRATSDataset(flair_dir=flair_dir,masks_dir=masks_dir, dataset_type="train", img_size=img_size)
    val_dataset = BRATSDataset(flair_dir=flair_dir,masks_dir=masks_dir, dataset_type="val", img_size=img_size)

    # Determine sizes for training and validation datasets
    train_size = len(train_dataset) if not train_size else train_size
    val_size = len(val_dataset) if not val_size else val_size

    # Generate list of indices
    train_indices = list(range(len(train_dataset)))
    val_indices = list(range(len(val_dataset)))

    # Create samplers for each split
    train_sampler = SubsetRandomSampler(train_indices[:train_size])
    val_sampler = SubsetRandomSampler(val_indices[:val_size])

    # Create data loaders using samplers
    train_loader = DataLoader(train_dataset, batch_size=batchsize, sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, sampler=val_sampler, num_workers=4, pin_memory=True, drop_last=True)

    return train_loader, val_loader

import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_flair_and_masks(data_loader, num_samples=3):
    """
    Visualizes FLAIR images and their corresponding masks from the DataLoader.

    :param data_loader: The DataLoader to sample data from.
    :param num_samples: Number of samples to display.
    """
    # Iterate through the data loader
    for i, batch in enumerate(data_loader):
        # Get the flair images and masks from the batch
        flair_images = batch['flair_image']  # Shape: [batch_size, 1, height, width] for grayscale
        masks = batch['masks']  # Shape: [batch_size, num_masks, 1, height, width]

        # Display a few samples (num_samples)
        for j in range(min(num_samples, flair_images.size(0))):
            flair_image = flair_images[j].cpu().numpy()  # Convert to numpy for plotting
            mask = masks[j].squeeze(1).cpu().numpy()  # Remove the singleton dimension (1) for masks

            # print(f"Sample {j + 1} - FLAIR Image Shape: {flair_image.shape}")
            # print(f"Sample {j + 1} - Mask Shape: {mask.shape}")

            # Plot the flair image and masks
            fig, axes = plt.subplots(1, len(mask) + 1, figsize=(12, 5))

            # Show the flair image (convert to [0, 1] range for visualization)
            axes[0].imshow(flair_image[0], cmap='gray')  # Show as grayscale (only one channel)
            axes[0].set_title("FLAIR Image")
            axes[0].axis('off')

            # Show the masks
            for m in range(mask.shape[0]):
                axes[m + 1].imshow(mask[m], cmap='gray')  # Show each mask in grayscale
                axes[m + 1].set_title(f"Mask {m + 1}")
                axes[m + 1].axis('off')

            plt.tight_layout()
            plt.show()

        # If we have already displayed the desired number of samples, stop iterating
        if i >= num_samples // flair_images.size(0):
            break


current_dir = os.getcwd()  # Get current working directory
flair_path = os.path.join(current_dir, "data/flair_images")
ensemble_path = os.path.join(current_dir, "data/ensemble_predictions")

# Example usage
train_loader, val_loader = get_data_loaders_BRATS(
    flair_dir=flair_path,
    masks_dir=ensemble_path,
    batchsize=2, img_size=(64, 64)
)
visualize_flair_and_masks(train_loader, num_samples=3)




'''
Training setup for colour(2chrominance channels) metrics:
- model takes 2 channel images as input.
- apply color-based transformations and spatial transformations.
- Upthe image size is a lower resolution(64x64) then the y-channels
- currently using a sparsity distance of 2; as well as looking at cross-channel correlations between cr and cb

The code below configures and trains a SUPN model for the colour part of the metric.
'''

## adapted from paula right now

import os
import torch
print(torch.cuda.is_available())
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

# from data_loader import get_data_loaders_BAAPS
from enum import Enum
import configparser
import wandb
import pdb

wandb.login()

class SupnBRATS:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Adjust according to your setup

        # Logging and data settings
        self.log_wandb = True  # Enable logging to wandb by default

        self.image_size = 64  # Image size set to 64 by default
        self.batch_size = 1  # Batch size set to 1 by default
        self.train_size = 5000  # Training data size set to 5000
        self.val_size = 200  # Validation data size set to 200
        self.data_loader_func = 'get_data_loaders_BRATS'  # Data loader function name (hardcoded)

        # Model settings
        self.local_connection_dist = [1]  # Local connection distance (hardcoded)
        self.num_levels = 5  # Number of levels in the model
        self.num_scales = 4  # Number of scales in the model
        self.num_pred_scales = 1  # Number of prediction scales
        self.dropout_rate = 0.0  # Dropout rate (no dropout)
        self.weight_decay = 0.0  # Weight decay set to 0 (no regularization)
        self.use_bias = False  # Do not use bias by default
        self.use_3D = False  # 3D operations are disabled by default
        self.max_ch = 1024  # Maximum number of channels in the model
        self.num_init_features = 32  # Number of initial features in the model
        # log diagonals needs to be 1 if 1 channel (mask 1,2,64,64 problem was caused by num_log_diag=2)
        self.num_log_diags = 1  # Number of log diagonals
        self.freeze_mean_decoder = True  # Freeze the mean decoder by default

        # File paths for saving and loading models
        self.supn_model_load_path = 'checkpoints/supn_model/colour1_lr0.0001_paramsnomean.pth'  # Path to load model
        self.supn_model_save_path = 'checkpoints/supn_model_ensebmle'  # Path to save the model

        # Optimizer settings
        self.optimizer_type = 'AdamW'  # Optimizer type set to AdamW

        os.makedirs(self.supn_model_save_path, exist_ok=True)

        self.train_schedules = [
            {
                'batch_size': 1,
                'learning_rate': 1e-3,
                'parameters': 'mean',
                'num_epochs': 1,
                'loss_type': 'mse_loss'
            },
            {
                'batch_size': 1,
                'learning_rate': 1e-4,
                'parameters': 'mean',
                'num_epochs': 1, # 2
                'loss_type': 'mse_loss'
            },
            {
                'batch_size': 1,
                'learning_rate': 1e-5,
                'parameters': 'nomean',
                'num_epochs': 5, #10
                'loss_type': 'log_likelihood_loss'
            },
            {
                'batch_size': 1,
                'learning_rate': 1e-5,
                'parameters': 'dec',
                'num_epochs': 2, # 5
                'loss_type': 'log_likelihood_loss'
            },
            {
                'batch_size': 1,
                'learning_rate': 1e-5,
                'parameters': 'chol',
                'num_epochs': 2, # 5
                'loss_type': 'log_likelihood_loss'
            },
            {
                'batch_size': 1,
                'learning_rate': 1e-6,
                'parameters': 'chol',
                'num_epochs': 2, # 5
                'loss_type': 'log_likelihood_loss'
            }
        ]

        if self.log_wandb:
            wandb.init(project="SUPNBraTs")


        self.step = 0
        self.epoch = 0
        self.eval_freq = 10

        self.model = SUPNEncoderDecoder(
            in_channels=1,
            num_out_ch=1,
            num_init_features=self.num_init_features,
            num_log_diags=self.num_log_diags,
            num_local_connections=self.local_connection_dist,
            use_skip_connections=True,
            num_scales=self.num_scales,
            num_prediction_scales=self.num_pred_scales,
            max_dropout=self.dropout_rate,
            use_bias=self.use_bias,
            use_3D=self.use_3D,
            max_ch=self.max_ch
        ).to(self.device)

        self.load_data()

        if self.supn_model_load_path and os.path.exists(self.supn_model_load_path):
            self.model.load_state_dict(torch.load(self.supn_model_load_path, map_location=self.device))


    def load_data(self):
        loader_func = eval(self.data_loader_func)
        self.train_dataloader, self.val_dataloader = loader_func(
            flair_path,
            ensemble_path,
            self.batch_size,
            (self.image_size, self.image_size),
        )

    # def __init__(self, flair_dir='dataset/2afc', masks_dir='', dataset_type="train", img_size=(256, 256), num_masks=5):


    def freeze_parameters(self, mode):
        if mode == 'mean':
            self._unfreeze_mean_decoder()
        if mode == 'nomean':
            self._freeze_mean_decoder()
        elif mode == 'dec':
            self._freeze_decoder()
        elif mode == 'chol':
            self._freeze_precision_decoder()
        elif mode == 'all':
            self._unfreeze_all()

    def _unfreeze_mean_decoder(self): # freezes everything but mean decoder, and encoder
        for param in self.model.parameters():
            param.requires_grad = True
        for param in self.model.encoder_decoder.supn_decoder.parameters():
            param.requires_grad = False

    def _freeze_mean_decoder(self): # freezes the mean decoder
        for param in self.model.parameters():
            param.requires_grad = True
        for param in self.model.encoder_decoder.mean_decoder.parameters():
            param.requires_grad = False

    def _freeze_decoder(self): # freezes everything(encodser,bottleneck) but the mean and supn decoder
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.encoder_decoder.mean_decoder.parameters():
            param.requires_grad = True
        for param in self.model.encoder_decoder.supn_decoder.parameters():
            param.requires_grad = True

    def _freeze_precision_decoder(self): # freezes everything but the choleski decoder
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.encoder_decoder.supn_decoder.parameters():
            param.requires_grad = True

    def _unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True


    def save_model(self, schedule):
        save_file = os.path.join(self.supn_model_save_path, f"{schedule}.pth")
        torch.save(self.model.state_dict(), save_file)
        print(f"Model saved to {save_file}")


    def get_losses_one_scale_supn(self, supn_data, target, level=0, only_mean=False):
        # assert isinstance(supn_data, SUPN)
        weighting_factor = 1.0 / (level + 1) * 5 # penelty based on level of transformation
        #print("Does supn_data.mean require gradients?", supn_data.mean.requires_grad)
        mse_loss_fn = nn.MSELoss(reduction='mean')
        # mse_loss = mse_loss_fn(supn_data.supn_data.mean.squeeze(0), target.to(self.device))
        # print(f"supn_data.mean shape: {supn_data.mean.shape}")
        # print(f"supn_data.mean squeeze shape: {supn_data.mean.squeeze(0).shape}")
        # print(f"target shape: {target.shape}")
        mse_loss = mse_loss_fn(supn_data.supn_data.mean.squeeze(0), target.to(self.device))
        mse_loss = mse_loss * weighting_factor

        # trying out hybrid loss
        bce_loss_fn = nn.BCEWithLogitsLoss()
        bce_loss = bce_loss_fn(supn_data.supn_data.mean.squeeze(0), target.to(self.device))

        combined_loss = 0.5*mse_loss + 0.5*bce_loss

        # before instead of combined it was just mse
  
        assert isinstance(mse_loss, torch.Tensor)

        if only_mean:
            return {'mse_loss': combined_loss}
        else:
            log_prob = -supn_data.log_prob(target.to(self.device)) * weighting_factor
            return {'mse_loss': combined_loss, 'log_likelihood_loss': log_prob.mean()}


    def run_model(self, image):
        image = image.to(self.device)
        return self.model(image)


    def run_epoch(self, loss_type):
        self.model.train()
        gradient_accumulation_steps = 10
        self.optimizer.zero_grad()

        accumulated_loss = 0.0  # logging

        for step, batch in enumerate(self.train_dataloader):
            loss = self.run_batch(batch, loss_type)[loss_type]
            loss = loss / gradient_accumulation_steps  #Normalize loss
            loss.backward()
            accumulated_loss += loss.item()  #logging

            if (step + 1) % gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                #print('accumulated',accumulated_loss)

                if self.log_wandb:
                    wandb.log({
                        f'{loss_type}': accumulated_loss,
                        'Step': self.step
                    })


                accumulated_loss = 0.0


    def run_batch(self, batch, loss_type='log_likelihood_loss'):
        loss_dict = {
            'mse_loss': torch.tensor(0.0, device=self.device),
            'log_likelihood_loss': torch.tensor(0.0, device=self.device)
        }

        flair_images = batch["flair_image"]  # Flair images for the batch
        masks = batch["masks"]  # Masks are the ensemble predictions (e.g., 3 masks per flair image)

        # print(f"Flair images shape: {flair_images.shape}")
        # print(f"Masks shape: {masks.shape}")

        for idx, flair_image in enumerate(flair_images):  # Loop through each flair image
            flair_image = flair_image.unsqueeze(0).to(self.device)  # Add batch dimension and move to device

            # print(f"Flair image shape: {flair_image.shape}")
            # print(f"Flair image unsqueeze shape: {flair_image.unsqueeze(0).shape}")
            # print(f"Masks shape: {masks.shape}")

            for mask_set in masks:  # Loop through each set of masks for this particular flair image
                mask = mask_set[idx].to(self.device)  # Get the corresponding mask for this flair image

                # Run the model on the flair image
                supn_outputs = self.run_model(flair_image)  # Generate model output for this flair image

                # Calculate the loss between the model output and the mask
                only_mean = loss_type != 'log_likelihood_loss'
                leveled_loss_dict = self.get_losses_one_scale_supn(
                    supn_outputs[0], mask, level=0, only_mean=only_mean
                )

                # Accumulate losses
                for loss_name, loss_value in leveled_loss_dict.items():
                    loss_dict[loss_name] += loss_value

        return loss_dict

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_dataloader:
                loss_dict = self.run_batch(batch, loss_type='log_likelihood_loss')
                if self.log_wandb:
                    if 'mse_loss' in loss_dict:
                        wandb.log({'Validation/mse_loss': loss_dict.get('mse_loss').item() , 'Step': self.step})
                    if 'log_likelihood_loss' in loss_dict:
                        wandb.log({'Validation/log_likelihood_loss': loss_dict.get('log_likelihood_loss'), 'Step': self.step})

    def set_optimizer(self, learning_rate):
        if self.optimizer_type == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")


    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))


    def train(self):
        for schedule_idx, schedule in enumerate(self.train_schedules):
            self.set_optimizer(schedule['learning_rate'])
            self.freeze_parameters(schedule['parameters'])
            print(f"Training with parameters: {schedule['parameters']}")
            for epoch in range(schedule['num_epochs']):
                self.run_epoch(schedule['loss_type'])
                # self.validate()
                print(epoch)
                stage_name = f"brats{schedule_idx}_lr{schedule['learning_rate']}_params{schedule['parameters']}"
                self.save_model(stage_name)
    
    def test(self):
        checkpoint_path = 'checkpoints/supn_model_ensebmle/brats5_lr1e-06_paramschol.pth'
        # 'checkpoints/supn_model/brats1_lr0.0001_paramsnomean.pth'
        self.load_model(checkpoint_path)
        self.model.eval()
        with torch.no_grad():
            for batch in self.train_dataloader:
                flair_images = batch["flair_image"]  # Flair images for the batch
                masks = batch["masks"]  # Masks are the ensemble predictions (e.g., 3 masks per flair image)

                # print(f"Flair images shape: {flair_images.shape}")
                # print(f"Masks shape: {masks.shape}")

                for idx, flair_image in enumerate(flair_images):  # Loop through each flair image
                    flair_image = flair_image.unsqueeze(0).to(self.device)  # Add batch dimension and move to device
                    supn_outputs = self.run_model(flair_image)
                    # supn_outputs = torch.sigmoid(supn_outputs)
                    mean  = torch.sigmoid(supn_outputs[0].mean)

                    # # Normalize the predicted mean and masks if necessary
                    # def normalize_image(image):
                    #     # Normalize the image to range [0, 1] for visualization
                    #     return (image - image.min()) / (image.max() - image.min())

                    # Step 1: Choose an image from the batch (e.g., the first one)
                    flair_image = flair_images[0]  # (channels, height, width)
                    mask = masks[idx] 
                    mask1 = mask[0].cpu().numpy()   # Remove the extra dimension (1, 64, 64) -> (64, 64)
                    mask2 = mask[1].cpu().numpy()   # Remove the extra dimension (1, 64, 64) -> (64, 64)
                    mask3 = mask[2].cpu().numpy()   # Remove the extra dimension (1, 64, 64) -> (64, 64)
                    # mask4 = mask[3].cpu().numpy()   # Remove the extra dimension (1, 64, 64) -> (64, 64)
                    mean_image = mean[0].squeeze().cpu().numpy()   # Mean, assuming it's a 2D or 3D array

                    # # Normalize images for better visualization
                    # flair_image = normalize_image(flair_image.cpu().numpy())
                    # mask = normalize_image(mask.cpu().numpy())
                    # mean_image = normalize_image(mean_image.cpu().numpy())

                    # Step 2: Create a grid with Flair Image, Mask, and Predicted Mean
                    fig, axes = plt.subplots(1, 5, figsize=(15, 5))

                    # Plot Flair Image
                    axes[0].imshow(flair_image[0], cmap='gray')  # Assuming the first channel is the desired one
                    axes[0].set_title("Flair Image")
                    axes[0].axis('off')

                    # Plot Mask(s)
                    axes[1].imshow(mask1[0], cmap='jet')  # Visualize the first mask (you could visualize all masks)
                    axes[1].set_title("Mask")
                    axes[1].axis('off')

                    axes[2].imshow(mask2[0], cmap='jet')  # Visualize the first mask (you could visualize all masks)
                    axes[2].set_title("Mask")
                    axes[2].axis('off')

                    axes[3].imshow(mask3[0], cmap='jet')  # Visualize the first mask (you could visualize all masks)
                    axes[3].set_title("Mask")
                    axes[3].axis('off')

                    # axes[4].imshow(mask4[0], cmap='jet')  # Visualize the first mask (you could visualize all masks)
                    # axes[4].set_title("Mask")
                    # axes[4].axis('off')

                    # Plot Predicted Mean
                    axes[4].imshow(mean_image, cmap='jet')  # Display predicted mean image
                    axes[4].set_title("Predicted Mean")
                    axes[4].axis('off')

                    # Step 3: Save the figure as an image
                    output_folder = "output_images"
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                    # Save the combined image
                    image_path = os.path.join(output_folder, f"flair_mask_predicted_mean_ensemb_{idx}.png")
                    plt.savefig(image_path, bbox_inches='tight', pad_inches=0.1)

                    # Optional: Close the plot to free memory
                    plt.close()

                    print(idx)

                    print(f"Combined image saved to {image_path}")

    # plt.use('Agg')
    def sample_validate_colour(self, nr_of_samples=10, log_wandb=False, train_set=False):
        import matplotlib.pyplot as plt
        from PIL import Image
        import torch

                # perceptual_model = SupnBRATS()
        # print('loadeed model')
        checkpoint_path = 'checkpoints/supn_model_ensebmle/brats5_lr1e-06_paramschol.pth'
        # 'checkpoints/supn_model/brats1_lr0.0001_paramsnomean.pth'
        self.load_model(checkpoint_path)
        self.model.eval()


        # def normalize(image_tensor):
        #         """Normalize the image tensor."""
        #         normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))
        #         return normalize(image_tensor)

        with torch.no_grad():
            for batch in self.train_dataloader:
                flair_images = batch["flair_image"]  # Flair images for the batch
                masks = batch["masks"]  # Masks are the ensemble predictions (e.g., 3 masks per flair image)

                # print(f"Flair images shape: {flair_images.shape}")
                # print(f"Masks shape: {masks.shape}")

                for idx, flair_image in enumerate(flair_images):  # Loop through each flair image
                    flair_image = flair_image.unsqueeze(0).to(self.device)  # Add batch dimension and move to device
                    supn_outputs = self.run_model(flair_image)
                    # supn_outputs = torch.sigmoid(supn_outputs)
                    mean  = torch.sigmoid(supn_outputs[0].mean)

                    flair_image = flair_images[0]  # (channels, height, width)




                    print(flair_image.size)
                    plt.figure(figsize=(6, 6))
                    plt.imshow(flair_image[0], cmap='gray') 
                    plt.axis('off')
                    plt.savefig('original_image.png')




                    # img1_tensor, c = preprocess_image(image, size=perceptual_model.image_size)
                    # print(img1_tensor.shape,img1_tensor.min().item(),img1_tensor.max().item())
                    # print(c.shape,c.min().item(),c.max().item())
                    # c = normalize(c)
                    # model_outputs = self.model.run_model(image)  # Get model outputs
                    # supn_list = model_outputs[0]

                    rows = nr_of_samples  # One row per sample
                    cols = 1  # One column per resolution level

                    plt.figure(figsize=(cols * 3, rows * 3))

                    # supn_dist = supn_list
                    supn_dist = supn_outputs[0]

                    # Sample from the distribution
                    sample = supn_dist.sample(num_samples=nr_of_samples).squeeze(1)
                    print("sample size", sample.shape)
                    print("sample min max", sample[0, 0, :, :].min(), sample[0, 0, :, :].max())
                    print("mean", mean.shape)
                    print("mean min max", mean.min(), mean.max())
                    sample_np = sample.detach().cpu().numpy()

                    fig1, axes = plt.subplots(2, 2, figsize=(10, 10))
                    vmin, vmax = -1, 1  # Set consistent color scaling limits
                    # mean = supn_dist.mean.detach().cpu()[:,0,:,:]
                    print(mean.shape,mean.min().item(),mean.max().item())
                    #print(image.min().item())
                    axes[0, 0].imshow(flair_image.squeeze(), cmap='gray')
                    axes[0, 0].set_title('Original Image')
                    axes[0, 0].axis('off')

                    # print(supn_dist.mean.detach().cpu().squeeze().shape)
                    # print(supn_dist.mean.detach().cpu().squeeze()[0].shape)

                    # mean  = torch.sigmoid(supn_dist.mean)


                    # axes[0, 1].imshow(supn_dist.mean.detach().cpu().squeeze()[0], cmap='gray')
                    axes[0, 1].imshow(mean.detach().cpu().squeeze(), cmap='gray')
                    axes[0, 1].set_title('Mean Reconstruction')
                    axes[0, 1].axis('off')

                    sample = torch.sigmoid(torch.from_numpy(sample_np[0, 0, :, :]))

                    # axes[1, 0].imshow(-supn_dist.mean.detach().cpu().squeeze()[0] + sample_np[0, 0, :, :], cmap='gray')
                    axes[1, 0].imshow(-mean.detach().cpu().squeeze() + sample.detach().cpu().squeeze(), cmap='gray')
                    axes[1, 0].set_title('Sampled Image (No Mean)')
                    axes[1, 0].axis('off')

                    axes[1, 1].imshow(sample.detach().cpu().squeeze(), cmap='gray')
                    axes[1, 1].set_title('Mean + Sample')
                    axes[1, 1].axis('off')

                    plt.tight_layout()
                    plt.show()

                    if log_wandb:
                        if train_set:
                            wandb.log({'Train_Recon': wandb.Image(fig1)})
                        else:
                            wandb.log({'Test_Recon': wandb.Image(fig1)})

                    plt.tight_layout()
                    plt.savefig(f'colour_sample_{idx}.png')


    # sample_validate_colour("data/flair_images/patient_0_4.png")



if __name__ == '__main__':
    trainer = SupnBRATS()
    # trainer.train()
    # trainer.test()
    trainer.sample_validate_colour()
