import torch
import matplotlib.pyplot as plt
import numpy as np
from notebooks.data_loader import get_data_loader
from u_net import UNet2D
from notebooks.data_loader import get_data_loader
import os

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

# Create validation data loader
val_loader = get_data_loader(
    batch_size=8,
    split="val",
    modality='flair',  # Assuming you're using the 'flair' modality
    mask_type='binary',  # Assuming binary segmentation
    resize=(64, 64)
)

# Create a directory to save predictions
output_dir = 'predictions'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save predictions for multiple models
def save_predictions(models, val_loader, num_images=5, output_dir='predictions'):
    images_saved = 0

    with torch.no_grad():  # No need to compute gradients for visualization
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            images, masks = images.unsqueeze(1), masks.unsqueeze(1)
            
            # Visualize and save the first 'num_images' predictions from multiple models
            for i in range(1):
                if images_saved >= num_images:
                    break

                img = images[i, 0, :, :].cpu().numpy()  # Convert the image to numpy for visualization
                true_mask = masks[i, 0, :, :].cpu().numpy()  # Convert the mask to numpy for visualization

                # Print the number of models to debug
                print(f"Number of models: {len(models)}")

                # Adjust the number of subplots to match the number of models + 2 (image and true mask)
                fig, ax = plt.subplots(1, len(models) + 2, figsize=(16, 4))  # +2 for image and ground truth

                # Display the original image and true mask first
                ax[0].imshow(img, cmap='gray')
                ax[0].set_title("Image")
                ax[1].imshow(true_mask, cmap='gray')
                ax[1].set_title("Ground Truth")

                # Iterate over models and their predictions
                for model_idx, model in enumerate(models):
                    outputs = model(images)
                    outputs = torch.sigmoid(outputs)  # Apply sigmoid for binary classification
                    pred_mask = outputs[i, 0, :, :].cpu().numpy()  # Convert the prediction to numpy

                    # Ensure the model_idx + 2 does not exceed the number of subplots
                    if model_idx + 2 < len(ax):
                        ax[model_idx + 2].imshow(pred_mask, cmap='gray')
                        ax[model_idx + 2].set_title(f"Model {model_idx + 1} Prediction")
                    else:
                        print(f"Skipping subplot for model {model_idx + 1} due to index overflow.")

                # Save the figure to the disk
                save_path = os.path.join(output_dir, f'pred_{images_saved+1}.png')
                plt.savefig(save_path)
                plt.close()  # Close the plot to avoid memory overload

                images_saved += 1
                print(f"Saved prediction {images_saved}/{num_images} at {save_path}")

# Save predictions from the validation set using multiple models


import torch
import numpy as np

def compute_model_variance(models, val_loader, num_images=5):
    # Store the predictions of each model for each image
    all_model_outputs = []

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            images, masks = images.unsqueeze(1), masks.unsqueeze(1)

            # For each image in the batch, get the predictions from each model
            model_preds = []
            for model in models:
                outputs = model(images)
                outputs = torch.sigmoid(outputs)  # Apply sigmoid for binary classification
                pred_mask = outputs.cpu().numpy()  # Convert to numpy for further processing
                model_preds.append(pred_mask)
            
            all_model_outputs.append(model_preds)

            if len(all_model_outputs) >= num_images:  # Stop after `num_images` images
                break

    # Compute the variance for each pixel across the models
    variance_per_pixel = []
    for image_idx in range(len(all_model_outputs)):  # Loop through images
        model_preds_for_image = np.array(all_model_outputs[image_idx])  # Shape: (num_models, height, width)
        
        # Calculate variance for each pixel
        variance = np.var(model_preds_for_image, axis=0)  # Compute variance along models axis
        
        variance_per_pixel.append(variance)

    # For simplicity, we can print the variance of the first image in the batch
    print("Variance of pixel values across models (for the first image):")
    print(variance_per_pixel[0])

    # You could also compute and print the mean of the variance across the entire image or batch
    mean_variance = np.mean(variance_per_pixel[0])  # Mean variance of all pixels in the first image
    print(f"Mean variance for the first image: {mean_variance:.4f}")

# Call the function to compute variance
# compute_model_variance(models, val_loader, num_images=5)

save_predictions(models, val_loader, num_images=15, output_dir=output_dir)
