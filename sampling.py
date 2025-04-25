import os
import time
import torch
import wandb
import matplotlib.pyplot as plt

def sample_model(trainer, nr_of_samples=10, log_wandb=False, train_set=False):
    """
    Samples predictions from a trained SUPN model, visualizes them, and optionally logs to Weights & Biases.

    This function performs inference using the `trainer` object's trained model. For each image in the 
    training dataset, it generates a mean prediction and multiple stochastic samples from the SUPN distribution. 
    It visualizes the original image, the predicted mean, the residual between a sample and the mean, and the 
    final sampled mask. The result can be shown, saved locally, and/or logged to WandB.

    Parameters:
    ----------
    trainer : SupnBRATS
        The trainer object that contains the model, dataloader, and configuration parameters.
    
    nr_of_samples : int, optional (default=10)
        Number of stochastic samples to draw from the model's predictive distribution.

    log_wandb : bool, optional (default=False)
        If True, logs the sampled images to Weights & Biases (wandb) for visualization and tracking.

    train_set : bool, optional (default=False)
        If True, logs the visualizations under 'Train_Recon' in wandb; otherwise logs as 'Test_Recon'.

    Returns:
    -------
    None
        Displays visualizations, saves them as .png files, and logs to wandb if enabled.
    """
    # Load the pre-trained model from checkpoint
    checkpoint_path = trainer.supn_model_load_path
    trainer.load_model(checkpoint_path)
    trainer.model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient computation for inference
        for batch in trainer.train_dataloader:
            flair_images = batch["modality_data"]  # Get Flair MRI images
            masks = batch["selected_pred"]  # Get ensemble masks (ground truth-like)

            for idx, flair_image in enumerate(flair_images):
                # Prepare input image for the model
                flair_image = flair_image.unsqueeze(0).to(trainer.device)  # Add batch dimension
                flair_image = flair_image.unsqueeze(1)  # Add channel dimension for grayscale

                # Forward pass through the model
                supn_outputs = trainer.run_model(flair_image)
                mean = torch.sigmoid(supn_outputs[0].mean)  # Apply sigmoid to model mean output

                # Just for visualization (grab the original input)
                flair_image_vis = flair_images

                # Save the original Flair image (optional step)
                plt.figure(figsize=(6, 6))
                plt.imshow(flair_image_vis[0], cmap='gray') 
                plt.axis('off')
                plt.savefig('original_image.png')

                # Sample from the learned distribution (multiple samples)
                supn_dist = supn_outputs[0]
                sample = supn_dist.sample(num_samples=nr_of_samples).squeeze(1)
                sample_np = sample.detach().cpu().numpy()

                # Apply sigmoid to convert logits to probabilities (for display)
                sample = torch.sigmoid(torch.from_numpy(sample_np[0, 0, :, :]))

                # Plot original image, mean, sampled difference, and sampled mask
                fig1, axes = plt.subplots(2, 2, figsize=(10, 10))

                # Original Flair image
                axes[0, 0].imshow(flair_image_vis.squeeze(), cmap='gray')
                axes[0, 0].set_title('Original Image')
                axes[0, 0].axis('off')

                # Predicted Mean output from the model
                axes[0, 1].imshow(mean.detach().cpu().squeeze(), cmap='gray')
                axes[0, 1].set_title('Mean Reconstruction')
                axes[0, 1].axis('off')

                # Residual sample without adding mean (uncertainty part)
                axes[1, 0].imshow(-mean.detach().cpu().squeeze() + sample.detach().cpu().squeeze(), cmap='gray')
                axes[1, 0].set_title('Sampled Image (No Mean)')
                axes[1, 0].axis('off')

                # Final Sample: mean + stochastic variation
                axes[1, 1].imshow(sample.detach().cpu().squeeze(), cmap='gray')
                axes[1, 1].set_title('Mean + Sample')
                axes[1, 1].axis('off')

                plt.tight_layout()
                plt.show()  # Display the grid of images

                # Optionally log the visualization to Weights & Biases
                if log_wandb:
                    image_log = {'Train_Recon' if train_set else 'Test_Recon': wandb.Image(fig1)}
                    wandb.log(image_log)

                # Save the composite image locally with a timestamp and patient ID
                timestamp = int(time.time())
                patient_id = batch["patient"][idx] if "patient" in batch else f"idx_{idx}"
                plt.savefig(f'drawn_sample_logit_{patient_id}_{timestamp}.png')
