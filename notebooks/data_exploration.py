import pandas as pd
from sklearn.model_selection import train_test_split

# # contains hgg and lgg meta data
# mapping_path = "/srv/thetis2/il221/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/name_mapping.csv"
# df = pd.read_csv(mapping_path) 

# # stratified splitting (LGG & HGG)
# train_data, temp_data = train_test_split(df, test_size=0.3, stratify=df["Grade"], random_state=42)
# val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data["Grade"], random_state=42)

# #save files
# train_data.to_csv("train_mapping.csv", index=False)
# val_data.to_csv("val_mapping.csv", index=False)
# test_data.to_csv("test_mapping.csv", index=False)


import pandas as pd

# Load the CSV files
train_data = pd.read_csv("train_mapping.csv")
val_data = pd.read_csv("val_mapping.csv")
test_data = pd.read_csv("test_mapping.csv")

# print("Train distribution:\n", train_data["Grade"].value_counts(normalize=True))
# print("Validation distribution:\n", val_data["Grade"].value_counts(normalize=True))
# print("Test distribution:\n", test_data["Grade"].value_counts(normalize=True))

print("Train distribution (counts):\n", train_data["Grade"].value_counts())
print("Validation distribution (counts):\n", val_data["Grade"].value_counts())
print("Test distribution (counts):\n", test_data["Grade"].value_counts())

