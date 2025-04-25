import os
import sys
import torch
import wandb

import torch.nn as nn
import torch.optim as optim

# Add the 'models' directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models')))

# Project-specific modules
from models.model import SUPNEncoderDecoder
from notebooks.data_loader import get_ensemble_all_data_loader
from sampling import sample_model

current_dir = os.getcwd()  # Get current working directory
flair_path = os.path.join(current_dir, "data/flair_images")
ensemble_path = os.path.join(current_dir, "data/ensemble_pred_logit")

data_dir="/home/data"

class SupnBRATS:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # logging and data settings
        self.log_wandb = True  # logging to wandb
        self.image_size = 64  # working with 64x64 images
        self.batch_size = 1  # batch size
        self.train_size = 1000  # train data set size
        self.val_size = 200  # val data set size
        self.data_loader_func = get_ensemble_all_data_loader  # hard coded data loader function

        # Model settings
        self.local_connection_dist = [1]  # Local connection distance (hardcoded)
        self.num_levels = 3  # num of levels in the model
        self.num_scales = 1  # num of scales in the model
        self.num_pred_scales = 1  # num of prediction scales
        self.dropout_rate = 0.0  # no dropout
        self.weight_decay = 0.0  # weight decay set to 0 (no regularization)
        self.use_bias = False  # no bias by default
        self.use_3D = False  # 3D operations are disabled by default
        self.max_ch = 128  # max number of channels in the model
        self.num_init_features = 32  # num of initial features in the model
        self.num_log_diags = 1  # num of log diagonals
        self.freeze_mean_decoder = True  # freezing the mean decoder

        # File paths for saving and loading models
        self.supn_model_load_path = 'check11/diag_only_lr0.0001_paramsnomean.pth'
        self.supn_model_save_path = 'check11/diag_only_run'

        # Optimizer settings
        self.optimizer_type = 'AdamW'  # Optimizer type set to AdamW

        os.makedirs(self.supn_model_save_path, exist_ok=True)

        self.train_schedules = [
            {
                'batch_size': 1,
                'learning_rate': 1e-4, # 3
                'parameters': 'mean',
                'num_epochs': 1,
                'loss_type': 'mse_loss'
            },
            {
                'batch_size': 1,
                'learning_rate': 1e-5, #4
                'parameters': 'mean',
                'num_epochs': 1, # 2
                'loss_type': 'mse_loss'
            },
            {
                'batch_size': 1,
                'learning_rate': 1e-5,
                'parameters': 'nomean',
                'num_epochs': 2, #10
                'loss_type': 'log_likelihood_loss'
            },
            {
                'batch_size': 1,
                'learning_rate': 1e-5,
                'parameters': 'dec',
                'num_epochs': 1, # 5
                'loss_type': 'log_likelihood_loss'
            },
            {
                'batch_size': 1,
                'learning_rate': 1e-5,
                'parameters': 'chol',
                'num_epochs': 1, # 5
                'loss_type': 'log_likelihood_loss'
            },
            {
                'batch_size': 1,
                'learning_rate': 1e-6,
                'parameters': 'chol',
                'num_epochs': 1, # 5
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
        assert callable(self.data_loader_func)

        self.train_dataloader = self.data_loader_func(
            data_dir=data_dir,
            batch_size=self.batch_size,
            split="train",
            modality='flair',
            mask_type='binary',
            resize=(self.image_size, self.image_size)
        )

        self.val_dataloader = self.data_loader_func(
            data_dir=data_dir,
            batch_size=self.batch_size,
            split="val",
            modality='flair',
            mask_type='binary',
            resize=(self.image_size, self.image_size)
        )

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
        weighting_factor = 1.0 / (level + 1) * 5
        mse_loss_fn = nn.MSELoss(reduction='mean')
        mse_loss = mse_loss_fn(supn_data.supn_data.mean.squeeze(0), target.to(self.device))

        # trying out hybrid loss
        bce_loss_fn = nn.BCEWithLogitsLoss()
        bce_loss = bce_loss_fn(supn_data.supn_data.mean.squeeze(0), target.to(self.device))

        combined_loss = 0.5*mse_loss + 0.5*bce_loss
  
        assert isinstance(mse_loss, torch.Tensor)

        # changed to mseloss
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

        accumulated_loss = 0.0 

        for step, batch in enumerate(self.train_dataloader):
            loss = self.run_batch(batch, loss_type)[loss_type]
            loss = loss / gradient_accumulation_steps  #Normalize loss
            loss.backward()

            accumulated_loss += loss.item() 

            if (step + 1) % gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

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

        flair_images = batch["modality_data"]  # Flair images for the batch
        masks = batch["selected_pred"]  # Masks are the ensemble predictions (e.g., 3 masks per flair image)

        for idx, flair_image in enumerate(flair_images):  # Loop through each flair image
            flair_image = flair_image.unsqueeze(0).to(self.device)  # Add batch dimension and move to device
            flair_image = flair_image.unsqueeze(1)  # Add channel dimension if it's grayscale

            masks = masks.permute(1, 0, 2, 3)

            for mask_set in masks:  # Loop through each set of masks for this particular flair image
                mask = mask_set[idx].to(self.device)  # Get the corresponding mask for this flair image
                mask = (mask > 0.5).float()

                mask = mask.unsqueeze(0)

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

    def sample(self, nr_of_samples=10, log_wandb=False, train_set=False):
        sample_model(self, nr_of_samples, log_wandb, train_set)


if __name__ == '__main__':
    wandb.login()
    trainer = SupnBRATS()
    trainer.train()
    # trainer.sample()
