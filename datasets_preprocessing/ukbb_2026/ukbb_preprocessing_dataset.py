import os
import json
import random
from pathlib import Path
import argparse

def create_dataset_json_ukbb(dataset_path, dataset_name, dataset_index, preprocessed_folder, split_ratio=0.8):

    image_path = Path(dataset_path) / "imagesTr"

    if not image_path.exists():
        raise FileNotFoundError(f"Image folder not found: {image_path}")

    images = [p for p in os.listdir(image_path) if p.endswith(".nii.gz")]
    print(f"Found {len(images)} images")

    subjects = {}

    for img in images:

        base = img.replace(".nii.gz", "")

        # Remove "_sax" suffix
        subject_id = base.replace("_sax", "")

        session_id = "sess0"
        modality = "MRI_cardiac_sax"

        subjects.setdefault(subject_id, {"sessions": {}})
        subjects[subject_id]["sessions"].setdefault(session_id, {"images": []})

        subjects[subject_id]["sessions"][session_id]["images"].append({
            "name": img,
            "image_path": str(image_path / img),
            "modality": modality,
            "associated_masks": None,
            "image_info": None
        })

    collection = {
        "collection_index": 1,
        "collection_name": dataset_name,
        "datasets": {
            str(dataset_index): {
                "dataset_index": dataset_index,
                "dataset_info": None,
                "name": f"Dataset{dataset_index}_{dataset_name}",
                "subjects": subjects
            }
        }
    }

    json_file = Path(dataset_path) / "pretrain_data.json"

    with open(json_file, "w") as f:
        json.dump(collection, f, indent=4)

    print(f"Saved dataset json to {json_file}")

    # Train / Val split
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

    print(f"Saved splits to {split_file}")
    print(f"Train: {len(train_subjects)} | Val: {len(val_subjects)}")

    return collection, splits


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--preprocessed_folder", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="UKBB_MRI")
    parser.add_argument("--dataset_index", type=int, default=701)

    args = parser.parse_args()

    create_dataset_json_ukbb(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        dataset_index=args.dataset_index,
        preprocessed_folder=args.preprocessed_folder
    )