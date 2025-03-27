import os
import numpy as np
import torch
from data_loader_test import get_data_loader
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
model_paths = [
    'ensemble_models/unet_model_1.pth', 
    'ensemble_models/unet_model_2.pth',
    'ensemble_models/unet_model_3.pth', 
    'ensemble_models/unet_model_4.pth', 
    'ensemble_models/unet_model_5.pth',
    'ensemble_models/unet_model_6.pth',
    'ensemble_models/unet_model_7.pth',
    'ensemble_models/unet_model_8.pth'
]

# Initialize models
models = [UNet2D(in_channels=1, out_channels=1).to(device) for _ in range(8)]
models = [load_model(model, model_path) for model, model_path in zip(models, model_paths)]

# # Training and Validation DataLoader
train_loader = get_data_loader(
    batch_size=8,
    split="train",
    modality='flair',  # Load only selected modalities
    mask_type='binary',  # Multi-class mask
    resize=(64, 64)
)

val_loader = get_data_loader(
    batch_size=8,
    split="val",
    modality='flair',  # Load only selected modalities
    mask_type='binary',  # Multi-class mask
    resize=(64, 64)
)

test_loader = get_data_loader(
    batch_size=8,
    split="test",
    modality='flair',  # Load only selected modalities
    mask_type='binary',  # Multi-class mask
    resize=(64, 64)
)


def save_predictions(models, data_loader, split='train', base_output_dir='/srv/thetis2/il221/BraTS2020_Processed'):
    """
    Saves model predictions for a given data loader (train or validation).
    
    Args:
        models (list): List of trained U-Net models.
        data_loader (DataLoader): DataLoader for either train or validation data.
        split (str): Either 'train' or 'val', used to define output directory.
        base_output_dir (str): Base directory where predictions will be stored.
    """
    output_dir = os.path.join(base_output_dir, split, 'ensemble_pred')
    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    images_saved = 0

    with torch.no_grad():
        for batch_idx, (images, masks, patient) in enumerate(data_loader):
            images = images.to(device).unsqueeze(1)  # Ensure shape [batch, 1, 64, 64]

            for i in range(images.shape[0]):  # Loop over batch

                # Generate predictions for all models
                ensemble_preds = []
                for model in models:
                    outputs = model(images)
                    outputs = torch.sigmoid(outputs)  # Apply sigmoid activation
                    pred_mask = outputs[i, 0, :, :].cpu().numpy()  # Convert to numpy
                    ensemble_preds.append(pred_mask)

                # Convert list to numpy array (Shape: [8, 64, 64])
                ensemble_preds = np.stack(ensemble_preds, axis=0)

                print(patient[i])

                # Define save path
                save_path = os.path.join(output_dir, patient[i])

                # Save as .npy
                np.save(save_path, ensemble_preds)

                print(f"Saved {split} prediction {images_saved+1} at {save_path}")
                images_saved += 1


# Save predictions for training data
save_predictions(models, train_loader, split='train')

# Save predictions for validation data
save_predictions(models, val_loader, split='val')

# Save predictions for validation data
save_predictions(models, test_loader, split='test')

