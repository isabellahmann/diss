# import os
# import numpy as np
# import random
# import matplotlib.pyplot as plt

# # Set the processed data directory
# data_dir = "/srv/thetis2/il221/BraTS2020_Processed"  # Update this with your actual path

# splits = ["train", "val", "test"]

# # Function to load a random patient and visualize a few slices
# def check_data(split="train", num_samples=3):
#     split_path = os.path.join(data_dir, split)
    
#     if not os.path.exists(split_path):
#         print(f"{split_path} does not exist!")
#         return
    
#     # Now we are looking inside each modality folder
#     modality_folders = [f for f in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, f))]
    
#     if not modality_folders:
#         print(f"No modality data found in {split_path}!")
#         return

#     # For each modality (flair, t1, etc.)
#     # for modality in modality_folders:
#     modality_path = os.path.join(split_path, "flair")

#     print(modality_path)
    
#     # List all files in the modality folder
#     flair_files = sorted([f for f in os.listdir(modality_path)])
#     mask_files = sorted([f for f in os.listdir(modality_path)])

#     if split == "test" or split == "val":
#             print(flair_files[0])


#     if len(flair_files) > 0:

#         sample_slices = random.sample(flair_files, min(num_samples, len(flair_files)))
        
        
#         for i, file in enumerate(sample_slices):
#             # Load flair and mask slices
#             flair_slice = np.load(os.path.join(modality_path, file))
#             mod_path = os.path.join(split_path, "seg")
#             mask_slice = np.load(os.path.join(mod_path, file)) 

#             fig, axes = plt.subplots(1, 2, figsize=(8,4))
            
#             axes[0].imshow(flair_slice, cmap="gray")
#             axes[0].set_title(f"Flair: {file}")
            
#             if mask_slice is not None:
#                 axes[1].imshow(mask_slice, cmap="gray")
#                 axes[1].set_title(f"Mask: {mask_slice}")
#             else:
#                 axes[1].axis("off")
#                 axes[1].set_title("No mask found")

#             base_filename = os.path.splitext(file)[0]
#             plt.tight_layout()
#             plt.savefig(f"{split}_{base_filename}")
#             plt.close() 
#     else:
#         print(f"Missing data {modality_path}.")

# # Run the check for each split
# for split in splits:
#     check_data(split=split)


import os
import numpy as np

def count_files_and_tumor_masks(base_dir):
    folder_counts = {}
    tumor_counts = {"train": {"tumor": 0, "no_tumor": 0}, 
                    "val": {"tumor": 0, "no_tumor": 0}, 
                    "test": {"tumor": 0, "no_tumor": 0}}

    # Check if base directory exists
    if not os.path.exists(base_dir):
        print(f"❌ Error: The directory {base_dir} does not exist!")
        return
    
    # Iterate over each split: 'train', 'val', 'test'
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(base_dir, split)
        
        if not os.path.exists(split_path):
            print(f"❌ Error: {split_path} does not exist!")
            continue
        
        # Count files in each modality folder
        for modality in ['flair', 'seg', 't1', 't1ce', 't2']:
            modality_path = os.path.join(split_path, modality)
            
            if os.path.exists(modality_path):
                file_count = len([f for f in os.listdir(modality_path) if os.path.isfile(os.path.join(modality_path, f))])
                folder_counts[f"{split}/{modality}"] = file_count
            else:
                folder_counts[f"{split}/{modality}"] = 0

        # Check segmentation masks (tumor vs. no tumor)
        seg_path = os.path.join(split_path, "seg")
        if os.path.exists(seg_path):
            for mask_file in os.listdir(seg_path):
                mask_file_path = os.path.join(seg_path, mask_file)
                
                # Load mask
                mask = np.load(mask_file_path).astype(np.int32)
                
                # Check if the mask contains any tumor pixels
                if np.any(mask > 0):
                    tumor_counts[split]["tumor"] += 1
                else:
                    tumor_counts[split]["no_tumor"] += 1

    return folder_counts, tumor_counts

# Base directory path (Change this to your specific dataset location)
base_directory = "/srv/thetis2/il221/BraTS2020_Processed"

# Run the function
folder_file_counts, tumor_counts = count_files_and_tumor_masks(base_directory)

# Print file counts
if folder_file_counts:
    print("\n📂 File counts in each folder:")
    for folder, count in folder_file_counts.items():
        print(f"{folder}: {count} files")

# Print tumor mask counts
print("\n🧠 Tumor vs. No-Tumor Mask Counts:")
for split, counts in tumor_counts.items():
    print(f"{split.upper()} - Tumor: {counts['tumor']} | No Tumor: {counts['no_tumor']}")

