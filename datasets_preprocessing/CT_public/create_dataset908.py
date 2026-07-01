cat > /mnt/all_data/ssl_foundation/data/create_dataset908.py << 'EOF'
#!/usr/bin/env python3
import os
import json
from pathlib import Path
import random

# ── Config ────────────────────────────────────────────────
CTRATE_TRAIN  = Path("/mnt/all_data/CT-RATE/dataset/train_fixed")
CTRATE_VALID  = Path("/mnt/all_data/CT-RATE/dataset/valid_fixed")
MERLIN_DIR    = Path("/mnt/all_data/foundation_ct/merlin_data")
FLARE_PART1   = Path("/mnt/all_data/Abdul/FLARE-Task4-CT-FM/train_part1")
FLARE_PART2   = Path("/mnt/all_data/Abdul/FLARE-Task4-CT-FM/train_part2")
OUTPUT_DIR    = Path("/mnt/all_data/ssl_foundation/data/nnssl_raw/Dataset908_CT")
DATASET_INDEX = 908
DATASET_NAME  = "Dataset908_CT"
SPLIT_RATIO   = 0.8
PROGRESS_EVERY = 5000
# ──────────────────────────────────────────────────────────

images_dir   = OUTPUT_DIR / "imagesTr"
preprocessed = OUTPUT_DIR / "preprocessed"
images_dir.mkdir(parents=True, exist_ok=True)
preprocessed.mkdir(parents=True, exist_ok=True)

subjects = {}
total_count = 0
skipped_duplicates = 0

def add_symlink(src_path: Path, images_dir: Path):
    symlink = images_dir / src_path.name
    if not symlink.exists():
        os.symlink(src_path.resolve(), symlink)
        return True
    return False

def add_flat_dataset(folder: Path, prefix: str, modality: str = "CT_noncontrast"):
    global total_count, skipped_duplicates
    files = sorted(folder.glob("*.nii.gz"))
    count = 0
    for f in files:
        subj_id = f"{prefix}_{f.stem}"
        sess_id = "sess0"
        subjects.setdefault(subj_id, {"sessions": {}})
        subjects[subj_id]["sessions"].setdefault(sess_id, {"images": []})
        ok = add_symlink(f, images_dir)
        if not ok:
            skipped_duplicates += 1
        subjects[subj_id]["sessions"][sess_id]["images"].append({
            "name": f.name,
            "image_path": str(f.resolve()),
            "modality": modality,
            "associated_masks": None,
            "image_info": None
        })
        count += 1
        total_count += 1
        if total_count % PROGRESS_EVERY == 0:
            print(f"  Progress: {total_count} files processed...")
    return count

def add_nested_dataset(folder: Path, prefix: str, modality: str = "CT_noncontrast"):
    global total_count, skipped_duplicates
    files = sorted(folder.rglob("*.nii.gz"))
    count = 0
    for f in files:
        rel = f.relative_to(folder)
        parts = rel.parts
        if len(parts) >= 2:
            subj_id = f"{prefix}_{parts[0]}"
            sess_id = parts[1] if len(parts) > 2 else "sess0"
        else:
            subj_id = f"{prefix}_{f.stem}"
            sess_id = "sess0"
        subjects.setdefault(subj_id, {"sessions": {}})
        subjects[subj_id]["sessions"].setdefault(sess_id, {"images": []})
        ok = add_symlink(f, images_dir)
        if not ok:
            skipped_duplicates += 1
        subjects[subj_id]["sessions"][sess_id]["images"].append({
            "name": f.name,
            "image_path": str(f.resolve()),
            "modality": modality,
            "associated_masks": None,
            "image_info": None
        })
        count += 1
        total_count += 1
        if total_count % PROGRESS_EVERY == 0:
            print(f"  Progress: {total_count} files processed...")
    return count

# ── 1. CT-RATE train ──────────────────────────────────────
print("Loading CT-RATE train...")
ct_train_count = add_nested_dataset(CTRATE_TRAIN, prefix="ctrate_train")
print(f"✅ CT-RATE train: {ct_train_count:>8,} files")

# ── 2. CT-RATE valid ──────────────────────────────────────
print("Loading CT-RATE valid...")
ct_valid_count = add_nested_dataset(CTRATE_VALID, prefix="ctrate_valid")
print(f"✅ CT-RATE valid: {ct_valid_count:>8,} files")

# ── 3. Merlin ─────────────────────────────────────────────
print("Loading Merlin...")
merlin_count = add_flat_dataset(MERLIN_DIR, prefix="merlin")
print(f"✅ Merlin:        {merlin_count:>8,} files")

# ── 4. FLARE part1 + part2 ───────────────────────────────
print("Loading FLARE part1...")
flare1_count = add_flat_dataset(FLARE_PART1, prefix="flare")
print(f"✅ FLARE part1:   {flare1_count:>8,} files")

print("Loading FLARE part2...")
flare2_count = add_flat_dataset(FLARE_PART2, prefix="flare")
print(f"✅ FLARE part2:   {flare2_count:>8,} files")

# ── 5. Save pretrain_data.json ────────────────────────────
print("\nSaving pretrain_data.json...")
collection = {
    "collection_index": 1,
    "collection_name": DATASET_NAME,
    "datasets": {
        str(DATASET_INDEX): {
            "dataset_index": DATASET_INDEX,
            "dataset_info": None,
            "name": DATASET_NAME,
            "subjects": subjects
        }
    }
}

json_file = OUTPUT_DIR / "pretrain_data.json"
with open(json_file, "w") as f:
    json.dump(collection, f, indent=4)
print(f"✅ Saved: {json_file}")

# ── 6. Train/val split ────────────────────────────────────
subject_ids = list(subjects.keys())
random.seed(42)
random.shuffle(subject_ids)
split_idx = int(SPLIT_RATIO * len(subject_ids))
train_ids = subject_ids[:split_idx]
val_ids   = subject_ids[split_idx:]

splits = {
    "train": [f"{DATASET_NAME}__{DATASET_INDEX}__{s}" for s in train_ids],
    "val":   [f"{DATASET_NAME}__{DATASET_INDEX}__{s}" for s in val_ids]
}

split_file = preprocessed / "splits_final.json"
with open(split_file, "w") as f:
    json.dump(splits, f, indent=4)

# ── 7. Final summary ──────────────────────────────────────
total_images = sum(
    len(sess["images"])
    for s in subjects.values()
    for sess in s["sessions"].values()
)

print(f"\n{'='*55}")
print(f"✅ Dataset908 created successfully!")
print(f"{'='*55}")
print(f"  CT-RATE train:    {ct_train_count:>8,}")
print(f"  CT-RATE valid:    {ct_valid_count:>8,}")
print(f"  Merlin:           {merlin_count:>8,}")
print(f"  FLARE part1:      {flare1_count:>8,}")
print(f"  FLARE part2:      {flare2_count:>8,}")
print(f"{'─'*55}")
print(f"  Total subjects:   {len(subjects):>8,}")
print(f"  Total images:     {total_images:>8,}")
print(f"  Skipped dupes:    {skipped_duplicates:>8,}")
print(f"  Train subjects:   {len(train_ids):>8,}")
print(f"  Val subjects:     {len(val_ids):>8,}")
print(f"  Output:           {OUTPUT_DIR}")
print(f"{'='*55}")
EOF

echo "Script written. Run with:"
echo "conda activate nnssl && python /mnt/all_data/ssl_foundation/data/create_dataset908.py"