#import os
#import json
#import random
#from pathlib import Path
#
## -------------------------------
## Configuration - edit if needed
## -------------------------------
#dataset_folder = "/mnt/all_data/ssl_foundation/data/nnssl_raw/Dataset701_UKBB_MRI"
#preprocessed_folder = "/mnt/all_data/ssl_foundation/data/nnssl_preprocessed/Dataset701_UKBB_MRI"
#dataset_index = 701
#dataset_name = "Dataset701_UKBB_MRI"
#split_ratio = 0.8
#modality = "MRI_cardiac_sax"
#session_id = "sess0"
## -------------------------------
#
#imagesTr = Path(dataset_folder) / "imagesTr"
#if not imagesTr.exists():
#    raise FileNotFoundError(f"imagesTr folder not found: {imagesTr}")
#
## List all NIfTI images
#images = [f for f in os.listdir(imagesTr) if f.endswith(".nii.gz")]
#print(f"Found {len(images)} images in {imagesTr}")
#
#subjects = {}
#
#for img in images:
#    # subject_id = filename without "_sax.nii.gz"
#    subject_id = img.replace("_sax.nii.gz", "")
#
#    subjects.setdefault(subject_id, {"sessions": {}})
#    subjects[subject_id]["sessions"].setdefault(session_id, {"images": []})
#    subjects[subject_id]["sessions"][session_id]["images"].append({
#        "name": img,
#        "image_path": str(imagesTr / img),  # ensure full path matches current folder
#        "modality": modality,
#        "associated_masks": None,
#        "image_info": None
#    })
#
## Build JSON structure
#collection = {
#    "collection_index": 1,
#    "collection_name": dataset_name,
#    "datasets": {
#        str(dataset_index): {
#            "dataset_index": dataset_index,
#            "dataset_info": None,
#            "name": dataset_name,
#            "subjects": subjects
#        }
#    }
#}
#
## Save pretrain_data.json
#json_file = Path(dataset_folder) / "pretrain_data.json"
#with open(json_file, "w") as f:
#    json.dump(collection, f, indent=4)
#print(f"Saved pretrain_data.json with {len(subjects)} subjects to {json_file}")
#
## -------------------------------
## Train/val split
## -------------------------------
#preprocessed_folder_path = Path(preprocessed_folder)
#preprocessed_folder_path.mkdir(parents=True, exist_ok=True)
#
#subject_ids = list(subjects.keys())
#random.seed(42)
#random.shuffle(subject_ids)
#split_idx = int(split_ratio * len(subject_ids))
#
#train_subjects = subject_ids[:split_idx]
#val_subjects = subject_ids[split_idx:]
#
#def format_id(sid):
#    return f"{dataset_name}__{dataset_index}__{sid}"
#
#splits = {
#    "train": [format_id(s) for s in train_subjects],
#    "val": [format_id(s) for s in val_subjects]
#}
#
#split_file = preprocessed_folder_path / "splits_final.json"
#with open(split_file, "w") as f:
#    json.dump(splits, f, indent=4)
#
#print(f"Saved splits_final.json to {split_file}")
#print(f"Train subjects: {len(train_subjects)} | Val subjects: {len(val_subjects)}")




############################################## 702 dataset #####################################

import os
import json
import random
from pathlib import Path

# -------------------------------
# Configuration
# -------------------------------
dataset_folder = "/mnt/all_data/ssl_foundation/data/nnssl_raw/Dataset702_Public_MRI"
preprocessed_folder = "/mnt/all_data/ssl_foundation/data/nnssl_preprocessed/Dataset702_Public_MRI"
dataset_index = 702
dataset_name = "Dataset702_Public_MRI"
split_ratio = 0.8
modality = "MRI_cardiac_sax"
session_id = "sess0"
# -------------------------------

imagesTr = Path(dataset_folder) / "imagesTr"
if not imagesTr.exists():
    raise FileNotFoundError(f"imagesTr folder not found: {imagesTr}")

# List all NIfTI images (skip labels)
images = [f for f in os.listdir(imagesTr) if f.endswith(".nii.gz") and "label" not in f.lower()]
print(f"Found {len(images)} images in {imagesTr}")

subjects = {}

for img in images:
    # Treat **each file as a unique subject**
    subject_id = img.replace(".nii.gz", "")

    subjects.setdefault(subject_id, {"sessions": {}})
    subjects[subject_id]["sessions"].setdefault(session_id, {"images": []})
    subjects[subject_id]["sessions"][session_id]["images"].append({
        "name": img,
        "image_path": str(imagesTr / img),
        "modality": modality,
        "associated_masks": None,
        "image_info": None
    })

# -------------------------------
# Build JSON structure
# -------------------------------
collection = {
    "collection_index": 1,
    "collection_name": dataset_name,
    "datasets": {
        str(dataset_index): {
            "dataset_index": dataset_index,
            "dataset_info": None,
            "name": dataset_name,
            "subjects": subjects
        }
    }
}

# Save pretrain_data.json
json_file = Path(dataset_folder) / "pretrain_data.json"
with open(json_file, "w") as f:
    json.dump(collection, f, indent=4)
print(f"Saved pretrain_data.json with {len(subjects)} subjects to {json_file}")

# -------------------------------
# Train/val split
# -------------------------------
preprocessed_folder_path = Path(preprocessed_folder)
preprocessed_folder_path.mkdir(parents=True, exist_ok=True)

subject_ids = list(subjects.keys())
random.seed(42)
random.shuffle(subject_ids)
split_idx = int(split_ratio * len(subject_ids))

train_subjects = subject_ids[:split_idx]
val_subjects = subject_ids[split_idx:]

def format_id(sid):
    return f"{dataset_name}__{dataset_index}__{sid}"

splits = {
    "train": [format_id(s) for s in train_subjects],
    "val": [format_id(s) for s in val_subjects]
}

split_file = preprocessed_folder_path / "splits_final.json"
with open(split_file, "w") as f:
    json.dump(splits, f, indent=4)

print(f"Saved splits_final.json to {split_file}")
print(f"Train subjects: {len(train_subjects)} | Val subjects: {len(val_subjects)}")























