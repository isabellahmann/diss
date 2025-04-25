import torch
import torch.nn.functional as F


def dice_coefficient(pred, target, eps=1e-8):
    """
    Computes the Dice coefficient between prediction and target masks.
    """
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + eps)


def iou_score(pred, target, eps=1e-8):
    """
    Computes the Intersection over Union (IoU).
    """
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return intersection / (union + eps)


def binary_cross_entropy(pred_probs, target):
    """
    Binary Cross Entropy loss for probabilistic outputs.
    """
    return F.binary_cross_entropy(pred_probs, target.float())


def mse_loss(pred, target):
    """
    Mean Squared Error between prediction and ground truth.
    """
    return F.mse_loss(pred, target.float())


def entropy_map(probs, eps=1e-8):
    """
    Computes the per-pixel entropy of a probability map.
    Input: probs shape (B, H, W) or (B, 1, H, W)
    """
    probs = torch.clamp(probs, eps, 1.0 - eps)
    entropy = -probs * torch.log(probs) - (1 - probs) * torch.log(1 - probs)
    return entropy


def variance_map(samples):
    """
    Computes the per-pixel variance across samples.
    Input: samples shape (N, H, W) or (B, N, H, W)
    """
    return samples.var(dim=0 if samples.dim() == 3 else 1)


def expected_calibration_error(probs, labels, n_bins=10):
    """
    Basic Expected Calibration Error (ECE) implementation.
    probs: predicted probabilities (flattened)
    labels: binary ground truth (flattened)
    """
    ece = 0.0
    probs = probs.detach().cpu()
    labels = labels.detach().cpu()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs > low) & (probs <= high)
        if mask.sum() > 0:
            avg_conf = probs[mask].mean()
            avg_acc = labels[mask].float().mean()
            ece += (mask.sum() / len(probs)) * torch.abs(avg_conf - avg_acc)

    return ece.item()


def compute_segmentation_metrics(probs, target, threshold=0.5, samples=None):
    """
    Aggregates several common segmentation metrics, optionally including uncertainty maps.
    
    Args:
        probs (Tensor): Predicted probabilities. Shape (H, W) or (B, 1, H, W)
        target (Tensor): Ground truth binary mask. Same shape.
        threshold (float): Threshold to binarize probs.
        samples (Tensor): Optional samples for uncertainty estimation (e.g., ensemble or posterior samples)

    Returns:
        dict: {dice, iou, bce, mse, entropy, variance}
    """
    probs = probs.squeeze()
    target = target.squeeze()

    pred_mask = (probs > threshold).float()

    metrics = {
        "dice": dice_coefficient(pred_mask, target).item(),
        "iou": iou_score(pred_mask, target).item(),
        "bce": binary_cross_entropy(probs, target).item(),
        "mse": mse_loss(probs, target).item(),
        "entropy": entropy_map(probs).mean().item()
    }

    if samples is not None:
        metrics["variance"] = variance_map(samples).mean().item()

    return metrics
