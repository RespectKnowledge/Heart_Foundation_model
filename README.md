
# Overview of proposed Foundation model
<img width="1011" height="1328" alt="Overall_diagram" src="https://github.com/user-attachments/assets/7d57335e-9a7d-4cd4-bdcc-9ebb8b6d5145" />
Heart Foundation Model

<img width="1011" height="1328" alt="Overall_diagram" src="https://github.com/user-attachments/assets/7d57335e-9a7d-4cd4-bdcc-9ebb8b6d5145" />
Cross-domain transfer learning for cardiac MRI: adapting a brain-pretrained DINOv3 + UxLSTM self-supervised encoder to downstream cardiac MRI tasks — segmentation, classification, and regression. Developed as part of a MICCAI 2026 submission.

Overview

This repository investigates whether self-supervised representations learned from brain MRI transfer effectively to cardiac MRI. The pipeline has two stages:


SSL Pretraining — a DINOv3-style self-distillation framework built on a UxLSTM backbone, pretrained on large-scale brain MRI (UK Biobank).
Downstream Adaptation — the pretrained encoder is fine-tuned (or probed) on cardiac MRI datasets for:

Segmentation (RV / myocardium / LV, via nnU-Net)
Classification (diagnosis, e.g. ACDC 5-class)
Regression (e.g. ejection fraction)





Repository Structure

Heart_Foundation_model/
+-- nnssl/                  # SSL pretraining framework (git submodule, fork of MIC-DKFZ/nnssl)
¦   +-- src/nnssl/training/nnsslTrainer/masked_image_modeling/
¦       +-- BaseDINOv3UxLSTMTrainer.py         # Stage 1: base pretraining
¦       +-- BaseDINOv3UxLSTMTrainerWithGram_701.py   # Stage 2: Gram-anchored refinement
¦       +-- BaseDINOv3UxLSTMTrainerHighRes_701.py    # Stage 3: high-resolution fine-tuning
¦
+-- Segmentation/           # Downstream cardiac segmentation (nnU-Net based)
¦   +-- UxLSTMHeartTrainer.py
¦   +-- EUPETrainer.py
¦   +-- BrainDINOv3Trainer.py
¦   +-- CMRTransformerTrainer.py
¦   +-- FlexiCTTrainer_NoPretrain.py
¦
+-- classification/         # Downstream classification / regression
¦   +-- uxlstm_cvd_heads.py     # Encoder + classification/regression heads
¦   +-- finetune_cmr.py         # Fine-tuning script (--task clf/reg, --mode probe/finetune)
¦
+-- Overall_diagram.png     # Pipeline overview figure
+-- README.md

SSL Pretraining (Stage 1–3)

Framework: nnssl (forked from MIC-DKFZ/nnssl), included here as a git submodule.


Stage 1 — BaseDINOv3UxLSTMTrainer: DINOv3 self-distillation with a UxLSTM (6-stage, features [32,64,128,256,320,320]) encoder, trained on UK Biobank brain MRI (Dataset701).
Stage 2 — BaseDINOv3UxLSTMTrainerWithGram_701: continues from the Stage 1 teacher checkpoint with a Gram-anchoring loss term.
Stage 3 — BaseDINOv3UxLSTMTrainerHighRes_701: high-resolution continuation for finer spatial detail.


Setup

bashgit clone --recurse-submodules https://github.com/RespectKnowledge/Heart_Foundation_model.git
cd Heart_Foundation_model/nnssl
pip install -e .

Training

bashnnssl_train 701 3d_fullres -tr BaseDINOv3UxLSTMTrainer
nnssl_train 701 3d_fullres -tr BaseDINOv3UxLSTMTrainerWithGram_701
nnssl_train 701 3d_fullres -tr BaseDINOv3UxLSTMTrainerHighRes_701

Downstream: Segmentation

Framework: custom fork of nnU-Net v2, with additional trainers for loading brain-pretrained SSL weights into a cardiac segmentation task.

Datasets: ACDC, M&Ms, M&Ms2, converted to nnU-Net format with labels bg/RV/MYO/LV = 0/1/2/3.

bashnnUNetv2_train <DATASET_ID> 3d_fullres <FOLD> -tr UxLSTMHeartTrainer

Evaluation uses Dice and HD95 (SimpleITK + scipy).

Downstream: Classification & Regression

uxlstm_cvd_heads.py loads the pretrained UxLSTM encoder and attaches a classification (LayerNorm + Dropout + Linear) or regression head on top of a pooled embedding.

finetune_cmr.py is a dataset-agnostic fine-tuning entrypoint:

bashpython finetune_cmr.py --task clf --mode probe --phases both --dataset acdc
python finetune_cmr.py --task reg --mode finetune --phases both --dataset acdc


--task : clf (classification) or reg (regression)
--mode : probe (frozen encoder) or finetune (full fine-tune)
--phases: ed, es, or both (dual-phase incorporates cardiac motion)


Datasets

DatasetRoleModalityUK Biobank Brain MRI (Dataset701)SSL pretrainingBrain MRIACDC (Dataset318)Downstream segmentation / classification / regressionCardiac cine MRIM&Ms, M&Ms2 (Dataset521)Downstream segmentationCardiac cine MRICMRxRecon2025Multi-label diagnosis benchmarkCardiac cine MRI

Environments

Separate conda environments are used per stage — see environments/ for exported specs:


nnssl — SSL pretraining
cardiac_downstream — segmentation fine-tuning
cinema — cardiac data preprocessing


Citation

bibtex@inproceedings{heart_foundation_model_2026,
  title     = {Cross-Domain Transfer from Brain to Cardiac MRI via Self-Supervised Pretraining},
  author    = {Qayyum, Abdul and collaborators},
  booktitle = {MICCAI},
  year      = {2026}
}

License

See LICENSE.md.

