import os
import numpy as np
import torch
# from other_data_loaders import get_data_loader
from u_net import UNet2D


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load a model checkpoint
def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

# # Load the trained U-Net model
# model = UNet2D(in_channels=1, out_channels=1).to(device)  # Initialize the model


# model_path = 'ensemble_models/unet_model_3.pth'  # Path to the saved model
# model = load_model(model, model_path)  # Load the saved model weights

# Load the trained U-Net models
# model_paths = [
#     'ensemble_models/unet_model_1.pth', 
#     'ensemble_models/unet_model_2.pth',
#     'ensemble_models/unet_model_3.pth', 
#     'ensemble_models/unet_model_4.pth', 
#     'ensemble_models/unet_model_5.pth',
#     'ensemble_models/unet_model_6.pth',
#     'ensemble_models/unet_model_7.pth',
#     'ensemble_models/unet_model_8.pth'
# ]

# # Initialize models
# models = [UNet2D(in_channels=1, out_channels=1).to(device) for _ in range(8)]
# models = [load_model(model, model_path) for model, model_path in zip(models, model_paths)]

# # # Training and Validation DataLoader
# train_loader = get_data_loader(
#     data_dir="/srv/thetis2/il221/BraTS2020_Processed",
#     batch_size=8,
#     split="train",
#     modality='flair',  # Load only selected modalities
#     mask_type='binary',  # Multi-class mask
#     resize=(64, 64)
# )

# val_loader = get_data_loader(
#     data_dir="/srv/thetis2/il221/BraTS2020_Processed",
#     batch_size=8,
#     split="val",
#     modality='flair',  # Load only selected modalities
#     mask_type='binary',  # Multi-class mask
#     resize=(64, 64)
# )

# test_loader = get_data_loader(
#     data_dir="/srv/thetis2/il221/BraTS2020_Processed",
#     batch_size=8,
#     split="test",
#     modality='flair',  # Load only selected modalities
#     mask_type='binary',  # Multi-class mask
#     resize=(64, 64)
# )


# def save_predictions(models, data_loader, split='train', base_output_dir='/srv/thetis2/il221/BraTS2020_Processed'):
#     """
#     Saves model predictions for a given data loader (train or validation).
    
#     Args:
#         models (list): List of trained U-Net models.
#         data_loader (DataLoader): DataLoader for either train or validation data.
#         split (str): Either 'train' or 'val', used to define output directory.
#         base_output_dir (str): Base directory where predictions will be stored.
#     """
#     output_dir = os.path.join(base_output_dir, split, 'ensemble_pred_logit')
#     os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#     images_saved = 0

#     with torch.no_grad():
#         for batch_idx, (images, masks, patient) in enumerate(data_loader):
#             images = images.to(device).unsqueeze(1)  # Ensure shape [batch, 1, 64, 64]

#             for i in range(images.shape[0]):  # Loop over batch

#                 # Generate predictions for all models
#                 ensemble_preds = []
#                 for model in models:
#                     outputs = model(images)
#                     # print(f"  Image shape: {outputs.shape}")
#                     # print(f"  Image range: [{outputs.min()}, {outputs.max()}]")
#                     # print(f"  Mean: {outputs.mean()}, Std: {outputs.std()}\n")  

#                     # outputs = torch.sigmoid(outputs)  # Apply sigmoid activation
#                     # print(f"  Sigmoid Image shape: {outputs.shape}")
#                     # print(f"  Image range: [{outputs.min()}, {outputs.max()}]")
#                     # print(f"  Mean: {outputs.mean()}, Std: {outputs.std()}\n") 

#                     pred_mask = outputs[i, 0, :, :].cpu().numpy()  # Convert to numpy

#                     # print(f"  Numpy Image shape: {pred_mask.shape}")
#                     # print(f"  Image range: [{pred_mask.min()}, {pred_mask.max()}]")
#                     # print(f"  Mean: {pred_mask.mean()}, Std: {pred_mask.std()}\n") 

#                     ensemble_preds.append(pred_mask)

#                 # Convert list to numpy array (Shape: [8, 64, 64])
#                 ensemble_preds = np.stack(ensemble_preds, axis=0)

#                 # print(patient[i])

#                 # Define save path
#                 # save_path = os.path.join(output_dir, patient[i])

#                 # # Save as .npy
#                 # np.save(save_path, ensemble_preds)

#                 # print(f"Saved {split} prediction {images_saved+1} at {save_path}")
#                 images_saved += 1


# Save predictions for training data
# save_predictions(models, train_loader, split='train')

# # Save predictions for validation data
# save_predictions(models, val_loader, split='val')

# # Save predictions for validation data
# save_predictions(models, test_loader, split='test')

# import os
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from data_loader_test import get_data_loader
# from u_net import UNet2D

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Function to load a model checkpoint
# def load_model(model, model_path):
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()  # Set the model to evaluation mode
#     return model

# # Load the trained U-Net models
# model_paths = [
#     'ensemble_models/unet_model_1.pth', 
#     'ensemble_models/unet_model_2.pth',
#     'ensemble_models/unet_model_3.pth', 
#     'ensemble_models/unet_model_4.pth', 
#     'ensemble_models/unet_model_5.pth',
#     'ensemble_models/unet_model_6.pth',
#     'ensemble_models/unet_model_7.pth',
#     'ensemble_models/unet_model_8.pth'
# ]

# # Initialize and load models
# models = [UNet2D(in_channels=1, out_channels=1).to(device) for _ in range(8)]
# models = [load_model(model, model_path) for model, model_path in zip(models, model_paths)]

# # Load Data
# train_loader = get_data_loader(
#     batch_size=8,
#     split="train",
#     modality='flair',  # Load only selected modalities
#     mask_type='binary',  # Multi-class mask
#     resize=(64, 64)
# )
# def save_debug_figures(models, data_loader, output_dir="debug_figures", num_samples=5):
#     """
#     Saves a few debug figures to check if predictions are correctly aligned with input images and ground truth.
    
#     Args:
#         models (list): List of trained U-Net models.
#         data_loader (DataLoader): DataLoader for test data.
#         output_dir (str): Directory where figures will be saved.
#         num_samples (int): Number of samples to save for inspection.
#     """
#     os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

#     with torch.no_grad():
#         for batch_idx, (images, gt_masks, patient) in enumerate(data_loader):
#             images = images.to(device)  # Ensure shape [batch, 1, 64, 64]
#             images = images.unsqueeze(1)
#             # Generate predictions for all models
#             batch_preds = []  # Stores all predictions for the batch
#             for model in models:
#                 outputs = model(images)  # Forward pass
#                 outputs = torch.sigmoid(outputs)  # Apply sigmoid activation
#                 batch_preds.append(outputs.cpu().numpy())  # Convert to numpy

#             # Convert list to numpy array (Shape: [8 models, batch_size, 64, 64])
#             batch_preds = np.stack(batch_preds, axis=0)

#             # Save only a few samples
#             for i in range(min(num_samples, images.shape[0])):
#                 fig, axes = plt.subplots(2, 5, figsize=(15, 6))

#                 # Input Image
#                 axes[0, 0].imshow(images[i, 0].cpu().numpy(), cmap='gray')
#                 axes[0, 0].set_title(f"Input Image {i+1}")
#                 axes[0, 0].axis('off')

#                 # Ground Truth Mask
#                 axes[0, 1].imshow(gt_masks[i].cpu().numpy(), cmap='gray')
#                 axes[0, 1].set_title(f"Ground Truth {i+1}")
#                 axes[0, 1].axis('off')

#                 # Model Predictions (Show 3 for variety)
#                 for j in range(3):
#                     pred = batch_preds[j, i]  # Model j’s prediction for image i
#                     pred = pred.squeeze()
#                     axes[0, j + 2].imshow(pred, cmap='jet')
#                     axes[0, j + 2].set_title(f"Model {j+1} Pred")
#                     axes[0, j + 2].axis('off')

#                 # Check for flipped predictions by showing vertical/horizontal flipped versions
#                 axes[1, 0].imshow(np.flipud(batch_preds[0, i].squeeze()), cmap='jet')  # Flip vertically
#                 axes[1, 0].set_title("Vertically Flipped Pred")
#                 axes[1, 0].axis('off')

#                 axes[1, 1].imshow(np.fliplr(batch_preds[0, i].squeeze()), cmap='jet')  # Flip horizontally
#                 axes[1, 1].set_title("Horizontally Flipped Pred")
#                 axes[1, 1].axis('off')

#                 # Save figure
#                 save_path = os.path.join(output_dir, f"sample_{batch_idx}_{i}.png")
#                 plt.tight_layout()
#                 plt.savefig(save_path)
#                 plt.close()

#                 print(f"Saved debug figure: {save_path}")

#             break  # Only process the first batch

# # Run function to save debug figures
# save_debug_figures(models, train_loader, num_samples=5)




# below is for testing if ensemble predictions are in logit space 

# import numpy as np



# base_directory = "/srv/thetis2/il221/BraTS2020_Processed"
# dir = "/srv/thetis2/il221/BraTS2020_Processed/train/ensemble_pred_logit"
# dir_ensemble = "/srv/thetis2/il221/BraTS2020_Processed/train/ensemble_pred"


# sample = "BraTS20_Training_274_slice_53.npy"


# # outside focker
# test_path = "/srv/thetis2/il221/BraTS2020_Processed/train/ensemble_pred/BraTS20_Training_274_slice_53.npy"

# # inside docker
# # data_dir="/home/data"
# test_path = "/home/data/train/ensemble_pred/BraTS20_Training_274_slice_53.npy"

# arr = np.load(test_path)
# print(arr.shape)         # Should be (8, 64, 64)
# print(arr.min(), arr.max())  # Should be outside [0, 1] if they're logits