"""
verify_and_combine_preprocessed.py

Step 1: Verify Dataset900 (Merlin) and Dataset902 (Inspect) preprocessed outputs are intact.
Step 2: Create a combined pretrain_data.json pointing to both preprocessed folders.
Step 3: (Optional) add Dataset907 (FLARE) with --add_flare flag.

Usage:
    # Verify + combine Merlin + Inspect only:
    python verify_and_combine_preprocessed.py

    # Verify + combine Merlin + Inspect + FLARE:
    python verify_and_combine_preprocessed.py --add_flare

    # Dry-run (verify only, don't write JSON):
    python verify_and_combine_preprocessed.py --dry_run
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

# ─── Paths ────────────────────────────────────────────────────────────────────
NNSSL_PREPROCESSED = Path("/mnt/all_data/ssl_foundation/data/nnssl_preprocessed")
NNSSL_RAW          = Path("/mnt/all_data/ssl_foundation/data/nnssl_raw")

DATASETS = {
    "900": {
        "name":       "Dataset900_MARLIN_CT",
        "raw_name":   "Dataset900_MARLIN_CT",      # folder under nnssl_raw (for pretrain_data.json ref)
        "label":      "merlin",
    },
    "902": {
        "name":       "Dataset902_Inspect",
        "raw_name":   "Dataset902_Inspect",
        "label":      "inspect",
    },
    "907": {
        "name":       "Dataset907_FLARE",
        "raw_name":   "Dataset907_FLARE",
        "label":      "flare",
        "optional":   True,
    },
}

OUTPUT_JSON = Path("/mnt/all_data/ssl_foundation/data/combined_900_902_pretrain.json")
OUTPUT_JSON_WITH_FLARE = Path("/mnt/all_data/ssl_foundation/data/combined_900_902_907_pretrain.json")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def verify_preprocessed_dataset(dataset_id: str, info: dict) -> dict:
    """Check preprocessed folder exists and count .b2nd files."""
    folder = NNSSL_PREPROCESSED / info["name"]
    result = {
        "dataset_id":   dataset_id,
        "folder":       str(folder),
        "exists":       folder.exists(),
        "b2nd_count":   0,
        "plans_exist":  False,
        "plans_file":   None,
        "sample_files": [],
        "ok":           False,
    }

    if not folder.exists():
        print(f"  ❌ [{dataset_id}] Folder missing: {folder}")
        return result

    # Count .b2nd files (preprocessed volumes)
    b2nd_files = list(folder.rglob("*.b2nd"))
    result["b2nd_count"] = len(b2nd_files)
    result["sample_files"] = [str(f) for f in b2nd_files[:3]]

    # Check for plans file
    plans_candidates = [
        folder / "nnsslPlans.json",
        folder / "nnsslPlans_onemmiso" / "nnsslPlans.json",
    ]
    for p in plans_candidates:
        if p.exists():
            result["plans_exist"] = True
            result["plans_file"] = str(p)
            break

    # Check subfolder structure
    subdirs = [d.name for d in folder.iterdir() if d.is_dir()]
    result["subdirs"] = subdirs

    result["ok"] = result["b2nd_count"] > 0
    return result


def load_raw_pretrain_json(dataset_id: str, info: dict) -> dict | None:
    """Load the raw pretrain_data.json for a dataset (to get subject/session metadata)."""
    raw_folder = NNSSL_RAW / info["raw_name"]
    json_path  = raw_folder / "pretrain_data.json"
    if not json_path.exists():
        print(f"  ⚠️  [{dataset_id}] No pretrain_data.json at {json_path}")
        return None
    with open(json_path) as f:
        return json.load(f)


def merge_pretrain_jsons(datasets_to_merge: list[tuple[str, dict, dict]]) -> dict:
    """
    Merge multiple pretrain_data.json dicts into one combined JSON.
    datasets_to_merge: list of (dataset_id, dataset_info, raw_json_dict)
    
    The combined JSON structure follows nnssl's expected format:
    {
        "datasets": {
            "900": { ...subjects... },
            "902": { ...subjects... },
        },
        "train": [...subject_ids...],
        "val":   [...subject_ids...],
    }
    """
    combined = {
        "datasets": {},
        "train": [],
        "val":   [],
    }

    for dataset_id, info, raw_json in datasets_to_merge:
        label = info["label"]

        # Extract subjects from this dataset's JSON
        # Handle both possible JSON structures
        if "datasets" in raw_json and dataset_id in raw_json["datasets"]:
            ds_block = raw_json["datasets"][dataset_id]
        elif "subjects" in raw_json:
            ds_block = raw_json
        else:
            # Try to find any top-level dict with subjects
            ds_block = None
            for key, val in raw_json.items():
                if isinstance(val, dict) and "subjects" in val:
                    ds_block = val
                    break

        if ds_block is None:
            print(f"  ⚠️  [{dataset_id}] Could not parse JSON structure — skipping")
            continue

        # Prefix subject IDs with dataset label to avoid collisions
        subjects_raw = ds_block.get("subjects", {})
        subjects_prefixed = {}
        for subj_id, subj_data in subjects_raw.items():
            new_id = f"{label}_{subj_id}"
            subjects_prefixed[new_id] = subj_data

        combined["datasets"][dataset_id] = {
            "name":     info["name"],
            "label":    label,
            "subjects": subjects_prefixed,
        }

        # Split train/val — use existing split if available, else re-split 90/10
        train_ids_raw = raw_json.get("train", [])
        val_ids_raw   = raw_json.get("val", [])

        if train_ids_raw or val_ids_raw:
            combined["train"].extend([f"{label}_{sid}" for sid in train_ids_raw])
            combined["val"].extend(  [f"{label}_{sid}" for sid in val_ids_raw])
        else:
            # No existing split — do 90/10
            all_ids = list(subjects_prefixed.keys())
            n_val   = max(1, int(len(all_ids) * 0.1))
            combined["val"].extend(all_ids[:n_val])
            combined["train"].extend(all_ids[n_val:])

    return combined


def print_summary(verify_results: dict, combined_json: dict | None = None):
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    total_b2nd = 0
    for ds_id, res in verify_results.items():
        status = "✅" if res["ok"] else "❌"
        print(f"\n{status} Dataset{ds_id}:")
        print(f"   Folder:      {res['folder']}")
        print(f"   .b2nd files: {res['b2nd_count']:,}")
        print(f"   Plans:       {'✅ ' + res['plans_file'] if res['plans_exist'] else '❌ missing'}")
        if res["sample_files"]:
            print(f"   Samples:     {res['sample_files'][0]}")
        if not res["ok"]:
            print(f"   ⚠️  PROBLEM: folder exists={res['exists']}, files={res['b2nd_count']}")
        total_b2nd += res["b2nd_count"]

    print(f"\n{'─'*60}")
    print(f"Total preprocessed volumes: {total_b2nd:,}")

    if combined_json:
        n_train = len(combined_json["train"])
        n_val   = len(combined_json["val"])
        n_ds    = len(combined_json["datasets"])
        print(f"\nCOMBINED JSON:")
        print(f"   Datasets:  {n_ds}")
        print(f"   Train:     {n_train:,}")
        print(f"   Val:       {n_val:,}")
        print(f"   Total:     {n_train + n_val:,}")
    print("="*60 + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--add_flare",  action="store_true", help="Include Dataset907 FLARE")
    parser.add_argument("--dry_run",    action="store_true", help="Verify only, do not write JSON")
    parser.add_argument("--output",     type=str,            help="Override output JSON path")
    args = parser.parse_args()

    # Decide which datasets to include
    active_ids = ["900", "902"]
    if args.add_flare:
        active_ids.append("907")

    print(f"\nDatasets to combine: {active_ids}")
    print(f"Add FLARE:           {args.add_flare}")
    print(f"Dry run:             {args.dry_run}\n")

    # ── Step 1: Verify preprocessed outputs ──────────────────────────────────
    print("STEP 1: Verifying preprocessed datasets...")
    verify_results = {}
    all_ok = True
    for ds_id in active_ids:
        info = DATASETS[ds_id]
        print(f"\n  Checking Dataset{ds_id} ({info['name']})...")
        result = verify_preprocessed_dataset(ds_id, info)
        verify_results[ds_id] = result
        if not result["ok"]:
            all_ok = False
            print(f"  ❌ Dataset{ds_id} has problems!")
        else:
            print(f"  ✅ Dataset{ds_id}: {result['b2nd_count']:,} .b2nd files found")

    if not all_ok:
        print("\n⚠️  Some datasets have issues. Check above. Continuing anyway to show what's available...")

    # ── Step 2: Load raw JSONs ────────────────────────────────────────────────
    print("\nSTEP 2: Loading raw pretrain_data.json files...")
    datasets_to_merge = []
    for ds_id in active_ids:
        info     = DATASETS[ds_id]
        raw_json = load_raw_pretrain_json(ds_id, info)
        if raw_json is None:
            print(f"  ⚠️  Skipping Dataset{ds_id} — no raw JSON")
            continue
        # Count subjects
        if "datasets" in raw_json and ds_id in raw_json["datasets"]:
            n_subj = len(raw_json["datasets"][ds_id].get("subjects", {}))
        elif "subjects" in raw_json:
            n_subj = len(raw_json["subjects"])
        else:
            n_subj = "?"
        print(f"  ✅ Dataset{ds_id}: {n_subj} subjects in JSON")
        datasets_to_merge.append((ds_id, info, raw_json))

    if not datasets_to_merge:
        print("❌ No datasets could be loaded. Exiting.")
        return

    # ── Step 3: Merge ─────────────────────────────────────────────────────────
    print("\nSTEP 3: Merging pretrain JSONs...")
    combined_json = merge_pretrain_jsons(datasets_to_merge)

    # ── Step 4: Write output ──────────────────────────────────────────────────
    if args.dry_run:
        print("\n[DRY RUN] Skipping JSON write.")
    else:
        if args.output:
            out_path = Path(args.output)
        elif args.add_flare:
            out_path = OUTPUT_JSON_WITH_FLARE
        else:
            out_path = OUTPUT_JSON

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(combined_json, f, indent=2)
        print(f"\n✅ Combined JSON written to: {out_path}")
        print(f"   Size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")

    # ── Step 5: Print summary ─────────────────────────────────────────────────
    print_summary(verify_results, combined_json)

    # ── Step 6: Print next steps ──────────────────────────────────────────────
    print("NEXT STEPS:")
    if not args.dry_run:
        if args.add_flare:
            print(f"  Use {OUTPUT_JSON_WITH_FLARE} for training")
        else:
            print(f"  Use {OUTPUT_JSON} for training")

    print("""
  To test nnssl training with the combined dataset:

    conda activate nnssl
    export nnssl_raw=/mnt/all_data/ssl_foundation/data/nnssl_raw
    export nnssl_preprocessed=/mnt/all_data/ssl_foundation/data/nnssl_preprocessed
    export nnssl_results=/mnt/all_data/ssl_foundation/data/nnssl_results

    # Quick smoke test (1 GPU, few steps):
    nnssl_train -d 900 902 -tr nnSSLTrainer --val_best -device cuda:0

    # Or with the merged JSON explicitly:
    nnssl_train --pretrain_json /mnt/all_data/ssl_foundation/data/combined_900_902_pretrain.json \\
                -tr nnSSLTrainer --val_best -device cuda:0
""")


if __name__ == "__main__":
    main()
