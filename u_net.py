import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
import os
import wandb 
from notebooks.other_data_loaders import get_data_loader

class UNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet2D, self).__init__()

        features = init_features
        self.encoder1 = UNet2D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet2D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet2D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet2D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet2D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet2D._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet2D._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet2D._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet2D._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
    

# Function to train a single U-Net model with W&B logging
def train_unet_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=100, patience=10, model_save_path=None):
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)


    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)  # Move to GPU
            # Add channel dimension to images: [8, 128, 128] -> [8, 1, 128, 128]
            images = images.unsqueeze(1)
            masks = masks.unsqueeze(1)

            optimizer.zero_grad()
            images = images.to(device)
            masks = masks.to(device).float()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Compute training loss for this epoch
        train_loss /= len(train_loader)

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)  # Move to GPU
                images = images.unsqueeze(1)
                masks = masks.unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        # Compute validation loss
        val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Log the metrics to W&B
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss
        })

        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save the model checkpoint
            if model_save_path:
                torch.save(model.state_dict(), model_save_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Function to train the ensemble of U-Nets with W&B logging
def train_ensemble_of_unets(ensemble_size, train_loader, val_loader, num_epochs=100, patience=10, save_dir='ensemble_models'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(ensemble_size):
        # Initialize a new model
        model = UNet2D(in_channels=1, out_channels=1).to(device)  # Move model to GPU
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # Define the model save path for each model in the ensemble
        model_save_path = os.path.join(save_dir, f'unet_model_{i+1}.pth')

        print(f"Training model {i+1}/{ensemble_size}")
        
        # Train the model and save it
        train_unet_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=num_epochs, patience=patience, model_save_path=model_save_path)


wandb.init(project="braTS-unet") 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ensemble_size = 8
criterion = BCEWithLogitsLoss()    

# load data
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

train_ensemble_of_unets(ensemble_size=1, train_loader=train_loader, val_loader=val_loader)

for images, masks in train_loader:
    print(images.shape)  
    images, masks = images.to(device), masks.to(device) 
