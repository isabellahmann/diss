import torch
from supn_test import SUPNEncoderDecoder  # Import your model class
import os
from supn_test import BRATSDataset, get_data_loaders_BRATS
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
        self.supn_model_load_path = 'checkpoints/supn_model/brats1_lr0.0001_paramsnomean.pth'  # Path to load model
        self.supn_model_save_path = 'checkpoints/supn_model'  # Path to save the model

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

        # if self.log_wandb:
        #     wandb.init(project="SUPNBraTs")


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
        current_dir = os.getcwd()  # Get current working directory
        flair_path = os.path.join(current_dir, "data/flair_images")
        ensemble_path = os.path.join(current_dir, "data/ensemble_predictions")
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
        assert isinstance(mse_loss, torch.Tensor)

        if only_mean:
            return {'mse_loss': mse_loss}
        else:
            log_prob = -supn_data.log_prob(target.to(self.device)) * weighting_factor
            return {'mse_loss': mse_loss, 'log_likelihood_loss': log_prob.mean()}


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

                # if self.log_wandb:
                #     wandb.log({
                #         f'{loss_type}': accumulated_loss,
                #         'Step': self.step
                #     })


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
                # if self.log_wandb:
                #     if 'mse_loss' in loss_dict:
                #         wandb.log({'Validation/mse_loss': loss_dict.get('mse_loss').item() , 'Step': self.step})
                #     if 'log_likelihood_loss' in loss_dict:
                #         wandb.log({'Validation/log_likelihood_loss': loss_dict.get('log_likelihood_loss'), 'Step': self.step})

    def set_optimizer(self, learning_rate):
        if self.optimizer_type == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")


    def load_model(self, path):
        self.model.load_state_dict(torch.load(path,map_location=self.device))

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
                    mean  = torch.sigmoid(supn_outputs[0].supn_data.mean)

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
                    image_path = os.path.join(output_folder, f"flair_mask_predicted_mean_ensem_{idx}.png")
                    plt.savefig(image_path, bbox_inches='tight', pad_inches=0.1)

                    # Optional: Close the plot to free memory
                    plt.close()

                    print(idx)

                    print(f"Combined image saved to {image_path}")

                        


if __name__ == '__main__':
    trainer = SupnBRATS()
    trainer.test()