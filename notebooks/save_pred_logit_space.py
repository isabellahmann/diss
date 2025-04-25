import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from u_net import UNet2D


# ----------------------------
# Dataset for flair-only input
# ----------------------------
class BraTSFlairOnlyDataset(Dataset):
    def __init__(self, data_dir, split='test', resize=(64, 64)):
        self.data_dir = os.path.join(data_dir, split, 'flair')
        self.resize = resize
        self.patients = sorted(os.listdir(self.data_dir))
        if not self.patients:
            raise ValueError(f"No data found in {self.data_dir}")

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient = self.patients[idx]
        flair_path = os.path.join(self.data_dir, patient)
        flair = np.load(flair_path).astype(np.float32)

        if self.resize:
            flair = cv2.resize(flair, self.resize, interpolation=cv2.INTER_LINEAR)

        flair = torch.tensor(flair.copy(), dtype=torch.float32)

        dummy_mask = torch.zeros_like(flair)
        return flair, dummy_mask, patient

# ----------------------------
# Ensemble prediction saver
# ----------------------------
def save_predictions(models, data_loader, split='train', base_output_dir='/home/data'):
    output_dir = os.path.join(base_output_dir, split, 'ensemble_pred_logit')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_saved = 0

    with torch.no_grad():
        for batch_idx, (images, _, patient_ids) in enumerate(data_loader):
            images = images.to(device).unsqueeze(1)  # [B, 1, H, W]

            for i in range(images.size(0)):
                ensemble_logits = []

                for model in models:
                    outputs = model(images)  # [B, 1, H, W]
                    logits = outputs[i, 0, :, :].cpu()  # shape: [H, W]
                    ensemble_logits.append(logits)

                stacked_logits = torch.stack(ensemble_logits)  # [8, H, W]

                patient_id = patient_ids[i].replace(".npy", "")
                save_path = os.path.join(output_dir, f"{patient_id}.npy")
                np.save(save_path, stacked_logits.numpy().astype(np.float32))

                print(f"[{split}] Saved: {save_path}")
                images_saved += 1


# ----------------------------
# Load ensemble models
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_paths = [f'ensemble_models/unet_model_{i}.pth' for i in range(1, 9)]

def load_model(model_path):
    model = UNet2D(in_channels=1, out_channels=1).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

models = [load_model(path) for path in model_paths]


# ----------------------------
# Load data loaders
# ----------------------------
def get_flair_only_loader(data_dir, split, batch_size=8, resize=(64, 64)):
    dataset = BraTSFlairOnlyDataset(data_dir=data_dir, split=split, resize=resize)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# data_dir = "/home/data"

# train_loader = get_flair_only_loader(data_dir, split='train')
# val_loader   = get_flair_only_loader(data_dir, split='val')
# test_loader  = get_flair_only_loader(data_dir, split='test')


# ----------------------------
# Save ensemble predictions
# ----------------------------

# dont want to rerun this right now 

# save_predictions(models, train_loader, split='train')
# save_predictions(models, val_loader, split='val')
# save_predictions(models, test_loader, split='test')


# ----------------------------
# Load and inspect one output
# ----------------------------

# data_dir="/home/data"

# logits = np.load('/home/data/train/ensemble_pred_logit/BraTS20_Training_274_slice_53.npy')
# logits = np.load('/home/data/test/ensemble_pred_logit/BraTS20_Training_357_slice_99.npy')

# print(logits.shape)       # should be [8, 64, 64]
# print(logits.min(), logits.max())  # should be ~[-10, 10]

