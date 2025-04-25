import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pandas as pd
from notebooks.data_loader import get_ensemble_all_data_loader

# Configuration
data_split = "train"  # Can be "train", "val", or "test"
output_dir = f"ensemble_eval_outputs/{data_split}"
os.makedirs(output_dir, exist_ok=True)

# Subdirectories for saving qualitative examples
success_dir = os.path.join(output_dir, "success_cases")
failure_dir = os.path.join(output_dir, "failure_cases")
os.makedirs(success_dir, exist_ok=True)
os.makedirs(failure_dir, exist_ok=True)

# CSV output path for storing quantitative results
csv_path = os.path.join(output_dir, "ensemble_metrics.csv")

# Load the dataloader
loader = get_ensemble_all_data_loader(
    data_dir="/home/data",
    batch_size=1,
    split=data_split,
    modality='flair',
    mask_type='binary',
    resize=(64, 64)
)

# Computes Dice, IoU, BCE loss, and uncertainty maps for ensemble predictions
def compute_ensemble_metrics(ensemble_logits, gt_mask, threshold=0.5):
    probs = torch.sigmoid(ensemble_logits)  # Convert logits to probabilities
    mean_prob = probs.mean(dim=0)  # Average over ensemble predictions
    var_map = probs.var(dim=0)     # Variance as uncertainty estimate
    pred_mask = (mean_prob > threshold).float()  # Binarized prediction

    gt_mask = (gt_mask > 0.5).float()  # Ensure binary ground truth

    # Flatten for metric computation
    pred_flat = pred_mask.view(-1)
    gt_flat = gt_mask.view(-1)

    intersection = (pred_flat * gt_flat).sum()
    dice = (2. * intersection) / (pred_flat.sum() + gt_flat.sum() + 1e-8)
    union = pred_flat.sum() + gt_flat.sum() - intersection
    iou = intersection / (union + 1e-8)
    bce = F.binary_cross_entropy(mean_prob, gt_mask)

    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "bce": bce.item(),
        "mean_prob": mean_prob.detach().cpu(),
        "var_map": var_map.detach().cpu(),
        "pred_mask": pred_mask.detach().cpu()
    }

# Evaluation loop with metric tracking and visualization
thresholds = {"success": 0.85, "failure": 0.65}
results = []

for batch in loader:
    flair = batch["modality_data"].squeeze(0)    # Flair MRI image
    gt_mask = batch["gt_mask"].squeeze(0)        # Ground truth mask
    ensemble_logits = batch["selected_pred"].squeeze(0)  # Ensemble model outputs
    patient = batch["patient"][0]  # Unique patient ID

    metrics = compute_ensemble_metrics(ensemble_logits, gt_mask)

    results.append({
        "Patient": patient,
        "Dice": metrics["dice"],
        "IoU": metrics["iou"],
        "BCE": metrics["bce"]
    })

    # Visual overlay of prediction vs. ground truth
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(flair.numpy(), cmap='gray')
    ax.contour(metrics["mean_prob"].numpy(), levels=[0.5], colors='r', linewidths=1.5)  # Predicted
    ax.contour(gt_mask.numpy(), levels=[0.5], colors='g', linewidths=1.5)  # Ground truth
    ax.axis('off')
    ax.set_title(f"{patient} (Dice={metrics['dice']:.3f})")

    red_patch = mpatches.Patch(color='red', label='Prediction')
    green_patch = mpatches.Patch(color='green', label='Ground Truth')
    ax.legend(handles=[red_patch, green_patch], loc='lower right', fontsize=8)

    # Save example based on performance threshold
    case_type = None
    if metrics["dice"] > thresholds["success"]:
        case_type = "success"
        out_path = os.path.join(success_dir, f"{patient}_overlay.png")
    elif metrics["dice"] < thresholds["failure"]:
        case_type = "failure"
        out_path = os.path.join(failure_dir, f"{patient}_overlay.png")
    if case_type:
        plt.savefig(out_path, bbox_inches='tight')
    plt.close()

# Create DataFrame of metrics for all patients
df = pd.DataFrame(results)

# Add rows for mean, std, and formatted mean ± std
mean_row = pd.DataFrame({"Patient": ["Mean"], "Dice": [df.Dice.mean()], "IoU": [df.IoU.mean()], "BCE": [df.BCE.mean()]})
std_row = pd.DataFrame({"Patient": ["Std"], "Dice": [df.Dice.std()], "IoU": [df.IoU.std()], "BCE": [df.BCE.std()]})
mean_std_row = pd.DataFrame({
    "Patient": ["Mean ± Std"],
    "Dice": [f"{df.Dice.mean():.4f} ± {df.Dice.std():.4f}"],
    "IoU": [f"{df.IoU.mean():.4f} ± {df.IoU.std():.4f}"],
    "BCE": [f"{df.BCE.mean():.4f} ± {df.BCE.std():.4f}"]
})

# Combine into one display-ready table
df_all = pd.concat([df, mean_row, std_row, mean_std_row], ignore_index=True)

# Save raw metrics to CSV (e.g., for LaTeX or spreadsheet use)
df.to_csv(csv_path, index=False)

# Save visual table as PNG
fig, ax = plt.subplots(figsize=(8, 0.4 * len(df_all) + 1))
ax.axis('off')
table = ax.table(cellText=df_all.values, colLabels=df_all.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
plt.title(f"Ensemble Segmentation Metrics ({data_split})", fontsize=12, pad=20)
img_path = os.path.join(output_dir, "ensemble_metrics_table.png")
plt.savefig(img_path, bbox_inches='tight')
plt.close()
