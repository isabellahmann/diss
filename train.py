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
from models.model import SUPNEncoderDecoder
from supn_base.supn_data import SUPNData
from supn_base.supn_distribution import SUPN
from data_loader import get_data_loaders_BAAPS
from enum import Enum
import configparser
import wandb


class SupnMetricColour:
    def __init__(self, config_path='supn_train_configs/supn_config_colour.ini'):
        # Load configurations
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

        self.device = torch.device(self.config.get('general', 'device', fallback='cuda'))
        self.log_wandb = self.config.getboolean('wandb', 'log_wandb', fallback=True)

        self.image_size = self.config.getint('data', 'image_size', fallback=64)
        self.batch_size = self.config.getint('training', 'batch_size', fallback=1)
        self.train_size = self.config.getint('data', 'train_size', fallback=5000)
        self.val_size = self.config.getint('data', 'val_size', fallback=200)
        self.data_loader_func = self.config.get('data', 'data_loader', fallback='get_data_loaders_BAAPS')

        self.local_connection_dist = eval(self.config.get('model', 'local_connection_dist', fallback="[3]"))
        self.num_levels = self.config.getint('model', 'num_levels', fallback=5)
        self.num_scales = self.config.getint('model', 'num_scales', fallback=1)
        self.num_pred_scales = self.config.getint('model', 'num_pred_scales', fallback=1)
        self.dropout_rate = self.config.getfloat('model', 'dropout_rate', fallback=0.0)
        self.weight_decay = self.config.getfloat('model', 'weight_decay', fallback=0.0)
        self.use_bias = self.config.getboolean('model', 'use_bias', fallback=False)
        self.use_3D = self.config.getboolean('model', 'use_3D', fallback=False)
        self.max_ch = self.config.getint('model', 'max_ch', fallback=1024)
        self.num_init_features = self.config.getint('model', 'num_init_features', fallback=32)
        self.num_log_diags = self.config.getint('model', 'num_log_diags', fallback=2)
        self.freeze_mean_decoder = self.config.getboolean('model', 'freeze_mean_decoder', fallback=True)


        self.supn_model_load_path = self.config.get('model', 'supn_model_load_path', fallback='checkpoints/colour_model/colour1_lr0.0001_paramsnomean.pth')
        self.supn_model_save_path = self.config.get('model', 'supn_model_save_path', fallback='checkpoints/colour_model')
        self.optimizer_type = self.config.get('training', 'optimiser', fallback='AdamW')



        os.makedirs(self.supn_model_save_path, exist_ok=True)

        self.train_schedules = []
        for section in self.config.sections():
            if section.startswith('train_schedule'):
                self.train_schedules.append({
                    'learning_rate': self.config.getfloat(section, 'learning_rate'),
                    'parameters': self.config.get(section, 'parameters'),
                    'num_epochs': self.config.getint(section, 'num_epochs'),
                    'loss_type': self.config.get(section, 'loss_type'),
                })

        if self.log_wandb:
            wandb.init(project="SUPNMetric", config=dict(self.config))
        

        self.step = 0
        self.epoch = 0
        self.eval_freq = 10

        self.model = SUPNEncoderDecoder(
            in_channels=2, 
            num_out_ch=2, 
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
            'data/dataset/2afc', 
            self.batch_size, 
            (self.image_size, self.image_size), 
            train_size=self.train_size, 
            val_size=self.val_size, 
            num_scales=self.num_pred_scales, 
            num_levels=self.num_levels,
            colour =True
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
        assert isinstance(supn_data, SUPN)
        weighting_factor = 1.0 / (level + 1) * 5 # penelty based on level of transformation
        #print("Does supn_data.mean require gradients?", supn_data.mean.requires_grad)
        mse_loss_fn = nn.MSELoss(reduction='mean')
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
        #image = image.unsqueeze(0)
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
        scales = batch["scales"]
        augmented_images = batch["augmented_images"]

        for level, results in enumerate(augmented_images):  # Loop through each sample in the batch
            only_mean = loss_type != 'log_likelihood_loss'
            supn_outputs = self.run_model(results[:,0,:,:])  # Get model output
            for i in range(results.shape[1]-1):  # Loop over the first dimension
                augmented_image = results.squeeze(0)[i+1]
                #print(f"Level {level}, Scale {res_level}:")
                #print(f"  Original Image Min: {results[:,0,:,:].min().item()}, Max: {results[:,0,:,:].max().item()}")
                #print(f"  Augmented Image Min: {augmented_image.min().item()}, Max: {augmented_image.max().item()}")

               
                leveled_loss_dict = self.get_losses_one_scale_supn(
                    supn_outputs[0], augmented_image, level, only_mean=only_mean
                )
                
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

    
    def load_model(self):
        self.model.load_state_dict(torch.load(self.supn_model_load_path))


    def train(self):
        for schedule_idx, schedule in enumerate(self.train_schedules):
            self.set_optimizer(schedule['learning_rate'])
            self.freeze_parameters(schedule['parameters'])
            for epoch in range(schedule['num_epochs']):
                self.run_epoch(schedule['loss_type'])
                self.validate()
                print(epoch)
                stage_name = f"colour{schedule_idx}_lr{schedule['learning_rate']}_params{schedule['parameters']}"
                self.save_model(stage_name)


if __name__ == '__main__':
    trainer = SupnMetricColour(config_path='configs/supn_config.ini')
    trainer.train()
