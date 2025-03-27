import os

path = "/srv/thetis2/il221/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"
old_name = path + "BraTS20_Training_355/W39_1998.09.19_Segm.nii"
new_name = path + "BraTS20_Training_355/BraTS20_Training_355_seg.nii"

# renaming the brats patient 355 file
try:
    os.rename(old_name, new_name)
    print("File re-named successfully!")
except:
    print("File already renamed!")