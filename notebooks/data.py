import os
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import train_test_split

data_dir = "/srv/thetis2/il221/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"
output_dir = "/srv/thetis2/il221/BraTS2020_Processed"
mapping_path = "/srv/thetis2/il221/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/name_mapping.csv"

modalities = ["flair", "t1", "t1ce", "t2", "seg"]

df = pd.read_csv(mapping_path)

train_data, temp_data = train_test_split(df, test_size=0.3, stratify=df["Grade"], random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data["Grade"], random_state=42)

# saving the patient splits
train_data.to_csv("train_mapping.csv", index=False)
val_data.to_csv("val_mapping.csv", index=False)
test_data.to_csv("test_mapping.csv", index=False)

# get patient ids
train_patients = set(train_data["BraTS_2020_subject_ID"])
val_patients = set(val_data["BraTS_2020_subject_ID"])
test_patients = set(test_data["BraTS_2020_subject_ID"])

for split in ['train', 'val', 'test']:
    for modality in modalities:
        os.makedirs(os.path.join(output_dir, split, modality), exist_ok=True)

for patient_folder in os.listdir(data_dir):
    patient_path = os.path.join(data_dir, patient_folder)
    if not os.path.isdir(patient_path):
        continue  # Skip non-folder files

    if patient_folder in train_patients:
        split = "train"
    elif patient_folder in val_patients:
        split = "val"
    elif patient_folder in test_patients:
        split = "test"
    else:
        continue  # Skip unknown patients

    # Process each modality
    for modality in modalities:
        nii_file = os.path.join(patient_path, f"{patient_folder}_{modality}.nii")
        if not os.path.exists(nii_file):
            print(f"Missing file: {nii_file}")
            continue

        # Load the 3D NIfTI file
        nii_img = nib.load(nii_file)
        img_data = nii_img.get_fdata()  # Shape: (H, W, D)

        # Normalize (except for segmentation masks)
        if modality != "seg":
            img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)

        # we only want slices starting from 30 to -30 for each scan
        selected_slices = img_data[:, :, 30:-30]  # Shape: (H, W, selected_depth)

        # save
        num_slices = selected_slices.shape[2]  # Updated depth
        for i in range(num_slices):
            slice_filename = f"{patient_folder}_slice_{i+30}.npy"  # Adjust slice numbering
            slice_path = os.path.join(output_dir, split, modality, slice_filename)
            np.save(slice_path, selected_slices[:, :, i])  # Save single slice

print("Finished saving files!")
