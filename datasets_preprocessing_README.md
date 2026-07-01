# Datasets Preprocessing

Scripts used to prepare raw imaging data into the format required for SSL pretraining (`nnssl`).

```
datasets_preprocessing/
├── CT_public/          # Preprocessing for public CT datasets
├── Cine_MRI_public/     # Preprocessing for public cardiac cine MRI datasets
└── ukbb_2026/           # Preprocessing for UK Biobank cardiac MRI (example below)
```

## Example: UK Biobank Cardiac MRI (`ukbb_2026/`)

1. **`preprocess_ukbb.py`** — converts raw UKBB cardiac data into per-frame NIfTI files (`{subject}_sax_frame00.nii.gz` … `frame49.nii.gz`), one file per cardiac cycle frame.
2. **`extract_all_frames.py`** — extracts all cine SAX frames per subject into `imagesTr/`.
3. **`create_dataset_json.py`** / **`ukbb_preprocessing_dataset.py`** — builds the `pretrain_data.json` manifest and an 80/20 **subject-level** train/val split (`splits_final.json`), ensuring no patient appears in both splits.
4. **`generate_ukbb_json.py`** — generates the final dataset JSON pointing to the preprocessed UKBB folder for `nnssl_train`.

```bash
python preprocess_ukbb.py
python extract_all_frames.py
python create_dataset_json.py --dataset_path <path> --preprocessed_folder <path> --dataset_index 703
```

## Other Datasets

`CT_public/` and `Cine_MRI_public/` follow the same overall pattern (raw → NIfTI conversion → dataset JSON → train/val split) for the additional public CT and cardiac MRI datasets used in pretraining. See the scripts in each folder for dataset-specific details.
