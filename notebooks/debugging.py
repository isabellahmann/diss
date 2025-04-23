import os
import numpy as np
import matplotlib.pyplot as plt

pred_path = "/srv/thetis2/il221/BraTS2020_Processed/train/ensemble_pred"
seg_path = "/srv/thetis2/il221/BraTS2020_Processed/train/seg"
flair_path = "/srv/thetis2/il221/BraTS2020_Processed/train/flair"

debug_dir = "debug"
os.makedirs(debug_dir, exist_ok=True)

for filename in os.listdir(pred_path):
    fig, axes = plt.subplots(3, figsize=(4, 8))  # Extra column for GT
    if filename.endswith(".npy"):  # Ensure it's an image file
        file_path = os.path.join(pred_path, filename)
        pred = np.load(file_path).astype(np.float32)
        seg = np.load(os.path.join(seg_path, filename)).astype(np.float32)
        flair = np.load(os.path.join(flair_path, filename)).astype(np.float32)


        print(f"Loaded: {filename}, Shape: {pred.shape}")

        # Display the image in the first column
        axes[0].imshow(flair, cmap='gray')
        axes[0].set_title(f"flair")
        axes[0].axis('off')

        # Display the ground truth mask in the second column
        axes[1].imshow(seg, cmap='gray')  # Use transparency for better visibility
        axes[1].set_title(f"GT Mask")
        axes[1].axis('off')

        axes[2].imshow(pred[0].squeeze(), cmap='gray')
        axes[2].set_title(f"Pred")
        axes[2].axis('off')


        save_path = f"debug/{filename.replace('.npy', '.png')}"
        # Save the figure
        plt.tight_layout()
        plt.savefig(save_path)



