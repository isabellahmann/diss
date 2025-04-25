import os
import numpy as np

def count_files_and_tumor_masks(base_dir):
    """
    Traverses a structured BraTS directory and:
    - Counts the number of .npy files in each modality subfolder.
    - Counts how many segmentation masks contain tumor vs. are empty.

    Args:
        base_dir (str): Path to the base directory containing train/val/test splits.

    Returns:
        folder_counts (dict): File counts per split/modality.
        tumor_counts (dict): Tumor/non-tumor counts per split.
    """
    folder_counts = {}
    tumor_counts = {split: {"tumor": 0, "no_tumor": 0} for split in ["train", "val", "test"]}

    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} does not exist.")
        return

    for split in ['train', 'val', 'test']:
        split_path = os.path.join(base_dir, split)
        if not os.path.exists(split_path):
            print(f"Warning: {split_path} not found. Skipping.")
            continue

        # Count files in each modality
        for modality in ['flair', 'seg', 't1', 't1ce', 't2']:
            modality_path = os.path.join(split_path, modality)
            if os.path.exists(modality_path):
                count = len([f for f in os.listdir(modality_path) if os.path.isfile(os.path.join(modality_path, f))])
                folder_counts[f"{split}/{modality}"] = count
            else:
                folder_counts[f"{split}/{modality}"] = 0

        # Analyze tumor masks
        seg_path = os.path.join(split_path, "seg")
        if os.path.exists(seg_path):
            for file in os.listdir(seg_path):
                mask = np.load(os.path.join(seg_path, file)).astype(np.int32)
                if np.any(mask > 0):
                    tumor_counts[split]["tumor"] += 1
                else:
                    tumor_counts[split]["no_tumor"] += 1

    return folder_counts, tumor_counts


# Run and display results
base_directory = "/srv/thetis2/il221/BraTS2020_Processed"
folder_counts, tumor_stats = count_files_and_tumor_masks(base_directory)

if folder_counts:
    print("File counts per modality:")
    for folder, count in folder_counts.items():
        print(f"{folder}: {count}")

print("Tumor vs No-Tumor mask counts:")
for split, stats in tumor_stats.items():
    print(f"{split.upper()}: Tumor = {stats['tumor']}, No Tumor = {stats['no_tumor']}")
