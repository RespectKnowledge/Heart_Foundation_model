# Heart Foundation Model

<img width="1011" height="1328" alt="Overall_diagram" src="https://github.com/user-attachments/assets/7d57335e-9a7d-4cd4-bdcc-9ebb8b6d5145" />

Cross-domain transfer learning for cardiac MRI: adapting a **brain-pretrained DINOv3 + UxLSTM self-supervised encoder** to downstream cardiac MRI tasks — segmentation, classification, and regression. 

## Overview

This repository investigates whether self-supervised representations learned  effectively from cardiac MRI and CT. The pipeline has two stages:

1. **SSL Pretraining** — a DINOv3-style self-distillation framework built on a UxLSTM backbone, pretrained on large-scale brain MRI and CT datasets.
2. **Downstream Adaptation** — the pretrained encoder is fine-tuned (or probed) on cardiac MRI and CT datasets for:
   - **Segmentation** (RV / myocardium / LV, Whole Heart CT segmentation, Whole heart Segmenntation with chambers, aretires, viens)
   - **Classification** (diagnosis, e.g. ACDC 5-class)
   - **Regression** (e.g. ejection fraction)

## Repository Structure

```
Heart_Foundation_model/
├── nnssl/                       # SSL pretraining framework (git submodule, fork of MIC-DKFZ/nnssl)
│   └── src/nnssl/training/nnsslTrainer/masked_image_modeling/
│       ├── BaseDINOv3UxLSTMTrainer.py         # Stage 1: base pretraining
│       ├── BaseDINOv3UxLSTMTrainerWithGram_701.py   # Stage 2: Gram-anchored refinement
│       └── BaseDINOv3UxLSTMTrainerHighRes_701.py    # Stage 3: high-resolution fine-tuning
│
├── Segmentation/                # Downstream cardiac segmentation
│   ├── UxLSTMHeartTrainer.py
│   ├── nnUNet
│   ├── Scar_Segmentation_models
│
├── classification/               # Downstream classification / regression
│   ├── uxlstm_cvd_heads.py       # Encoder + classification/regression heads
│   └── finetune_cmr.py           # Fine-tuning script (--task clf/reg, --mode probe/finetune)
│
├── datasets_preprocessing/       # Data preparation scripts for SSL pretraining sources
│   ├── CT_public/                 # Public CT dataset preprocessing
│   ├── Cine_MRI_public/           # Public cardiac cine MRI dataset preprocessing
│   └── ukbb_2026/                 # UK Biobank cardiac MRI preprocessing
│
├── Overall_diagram.png           # Pipeline overview figure
└── README.md
```

## SSL Pretraining (Stage 1–3)

Framework: [`nnssl`](https://github.com/RespectKnowledge/nnssl), included here as a git submodule.

- **Stage 1** — `BaseDINOv3UxLSTMTrainer`: DINOv3 self-distillation with a UxLSTM (6-stage, features `[32,64,128,256,320,320]`) encoder, trained on CT and MRI datasets.
- **Stage 2** — `BaseDINOv3UxLSTMTrainerWithGram_701`: continues from the Stage 1 teacher checkpoint with a Gram-anchoring loss term.
- **Stage 3** — `BaseDINOv3UxLSTMTrainerHighRes_701`: high-resolution continuation for finer spatial detail.

### Setup

```bash
git clone --recurse-submodules https://github.com/RespectKnowledge/Heart_Foundation_model.git
cd Heart_Foundation_model/nnssl
pip install -e .
```

### Training

```bash
nnssl_train 701 3d_fullres -tr BaseDINOv3UxLSTMTrainer
nnssl_train 701 3d_fullres -tr BaseDINOv3UxLSTMTrainerWithGram_701
nnssl_train 701 3d_fullres -tr BaseDINOv3UxLSTMTrainerHighRes_701
```

## Datasets Preprocessing

See [`datasets_preprocessing/`](datasets_preprocessing/) for scripts that prepare raw data into the format required for SSL pretraining, covering:

- **`ukbb_2026/`** — UK Biobank cardiac MRI: raw → per-frame NIfTI → dataset JSON → subject-level train/val split. See [`datasets_preprocessing/README.md`](datasets_preprocessing/README.md) for a worked example.
- **`CT_public/`** — public CT dataset preprocessing (same overall pattern).
- **`Cine_MRI_public/`** — public cardiac cine MRI dataset preprocessing (same overall pattern).

## Downstream: Segmentation

Framework: custom fork of [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet), with additional trainers for loading brain-pretrained SSL weights into a cardiac segmentation task.

Datasets: ACDC, converted to nnU-Net format with labels `bg/RV/MYO/LV = 0/1/2/3`.

```bash
nnUNetv2_train <DATASET_ID> 3d_fullres <FOLD> -tr UxLSTMHeartTrainer
```

Evaluation uses Dice and HD95 (SimpleITK + scipy).

## Downstream: Classification & Regression

`uxlstm_cvd_heads.py` loads the pretrained UxLSTM encoder and attaches a classification (`LayerNorm + Dropout + Linear`) or regression head on top of a pooled embedding.

`finetune_cmr.py` is a dataset-agnostic fine-tuning entrypoint:

```bash
python finetune_cmr.py --task clf --mode probe --phases both --dataset acdc
python finetune_cmr.py --task reg --mode finetune --phases both --dataset acdc
```

- `--task` : `clf` (classification) or `reg` (regression)
- `--mode` : `probe` (frozen encoder) or `finetune` (full fine-tune)
- `--phases`: `ed`, `es`, or `both` (dual-phase incorporates cardiac motion)

## Datasets

| Dataset | Role | Modality |
|---|---|---|
| UK Biobank Brain MRI (Dataset701) | SSL pretraining | Cardiac Cine MRI |
| ACDC (Dataset318) | Downstream segmentation / classification / regression | Cardiac cine MRI |

## Environments

Separate conda environments are used per stage — see `environments/` for exported specs:
- `nnssl` — SSL pretraining
- `cardiac_downstream` — segmentation fine-tuning
- `cinema` — cardiac data preprocessing

## Citation

```bibtex
@misc{castlefm_github,
  author       = {Qayyum, Abdul and Mazher, Moona and Ugurlu, Devran and Solis Lemus, Jose Alonso and Rodero, Cristobal and Sillett, Charles and Lefebvre, Arthur and Niederer, Steven A.},
  title        = {{CASTLE-FM}: Towards a Multimodal Cardiac Foundation Model for Generalizable Cardiovascular Image Analysis},
  year         = {2026},
  howpublished = {\url{https://github.com/RespectKnowledge/Heart_Foundation_model}},
  note         = {GitHub repository}
}
```

## License

See `LICENSE.md`.
