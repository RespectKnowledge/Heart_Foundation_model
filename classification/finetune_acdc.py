"""
Fine-tune the UxLSTM (DINOv3+UxLSTM SSL) encoder on ACDC (CineMA-processed layout).

Two tasks:
  --task clf : 5-class diagnosis (DCM / HCM / MINF / NOR / RV)  [CrossEntropy]
  --task reg : LV ejection fraction regression (lv_ef)          [SmoothL1]

Two modes:
  --mode probe    : encoder frozen, train head only (fast, best with few patients)
  --mode finetune : unfreeze encoder, low LR (run AFTER you have a probe baseline)

Reuses model definitions from uxlstm_cvd_heads.py (same folder).

Data layout (CineMA processed):
    ROOT/train/patient001/patient001_sax_ed.nii.gz   (+ _es, _gt, _t)
    ROOT/train_metadata.csv   columns: pid,pathology,...,lv_ef,...
    ROOT/test/...   ROOT/test_metadata.csv
Input volume = the ED short-axis stack (patientXXX_sax_ed.nii.gz).
Labels are read straight from the CSV.
"""

import argparse
import os
import csv
import random
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from uxlstm_cvd_heads import load_pretrained_bot, UxLSTMClassifier, UxLSTMRegressor

# ── config ──────────────────────────────────────────────────────────────────────
ROOT      = "/mnt/all_data/Abdul/Foundation_MRI_datasets/Cine_MRI/CineMA/data/acdc/processed"
PATCH     = (128, 128, 128)
TARGET_SP = (1.0, 1.0, 1.0)          # onemmiso
CLASSES   = ["DCM", "HCM", "MINF", "NOR", "RV"]
SEED      = 42


# ── preprocessing (match SSL) ───────────────────────────────────────────────────
def resample_iso(img, target_sp=TARGET_SP):
    in_sp, in_sz = img.GetSpacing(), img.GetSize()
    out_sz = [int(round(in_sz[i] * in_sp[i] / target_sp[i])) for i in range(3)]
    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing(target_sp)
    rs.SetSize(out_sz)
    rs.SetOutputDirection(img.GetDirection())
    rs.SetOutputOrigin(img.GetOrigin())
    rs.SetInterpolator(sitk.sitkBSpline)
    return rs.Execute(img)


def crop_or_pad(arr, patch=PATCH):
    out = np.zeros(patch, dtype=arr.dtype)
    src, dst = [], []
    for i in range(3):
        n, p = arr.shape[i], patch[i]
        if n >= p:
            s = (n - p) // 2
            src.append(slice(s, s + p)); dst.append(slice(0, p))
        else:
            s = (p - n) // 2
            src.append(slice(0, n)); dst.append(slice(s, s + n))
    out[dst[0], dst[1], dst[2]] = arr[src[0], src[1], src[2]]
    return out


def load_volume(path):
    arr = sitk.GetArrayFromImage(resample_iso(sitk.ReadImage(path))).astype(np.float32)
    m, s = arr.mean(), arr.std()
    arr = (arr - m) / (s + 1e-8)
    return crop_or_pad(arr)


def read_metadata(csv_path):
    """pid -> row dict."""
    rows = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows[r["pid"]] = r
    return rows


# ── dataset ──────────────────────────────────────────────────────────────────────
class ACDCDataset(Dataset):
    def __init__(self, split_dir, meta, pids, task="clf", augment=False):
        self.task, self.augment = task, augment
        self.items = []
        for pid in pids:
            img = os.path.join(split_dir, pid, f"{pid}_sax_ed.nii.gz")
            if not os.path.exists(img):
                continue
            row = meta[pid]
            if task == "clf":
                if row["pathology"] not in CLASSES:
                    continue
                target = CLASSES.index(row["pathology"])
            else:
                target = float(row["lv_ef"])
            self.items.append((img, target))
        print(f"  {task} set: {len(self.items)} cases")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, target = self.items[i]
        vol = load_volume(path)
        if self.augment:
            for ax in range(3):
                if random.random() < 0.5:
                    vol = np.flip(vol, ax).copy()
            vol = vol * random.uniform(0.9, 1.1)
        x = torch.from_numpy(vol).unsqueeze(0).float()
        if self.task == "clf":
            return x, torch.tensor(target, dtype=torch.long)
        return x, torch.tensor([target], dtype=torch.float32)


# ── stratified train/val split over the 100 train patients ───────────────────────
def train_val_pids(meta, val_frac=0.2):
    by_group = {}
    for pid, row in meta.items():
        by_group.setdefault(row["pathology"], []).append(pid)
    rng = random.Random(SEED)
    train, val = [], []
    for g, ps in by_group.items():
        ps = sorted(ps); rng.shuffle(ps)
        k = max(1, int(round(len(ps) * val_frac)))
        val += ps[:k]; train += ps[k:]
    return train, val


# ── metrics ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, task, device):
    model.eval()
    preds, gts = [], []
    for x, y in loader:
        out = model(x.to(device))
        preds.append((out.argmax(1) if task == "clf" else out).cpu())
        gts.append(y)
    preds = torch.cat(preds).numpy(); gts = torch.cat(gts).numpy()
    if task == "clf":
        acc = (preds == gts).mean()
        baccs = [(preds[gts == c] == c).mean() for c in np.unique(gts)]
        return {"acc": float(acc), "bal_acc": float(np.mean(baccs))}
    mae = np.abs(preds - gts).mean()
    ss_res = ((preds - gts) ** 2).sum()
    ss_tot = ((gts - gts.mean()) ** 2).sum() + 1e-8
    return {"mae": float(mae), "r2": float(1 - ss_res / ss_tot)}


# ── train ────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--task", choices=["clf", "reg"], default="clf")
    ap.add_argument("--mode", choices=["probe", "finetune"], default="probe")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"uxlstm_acdc_{args.task}_{args.mode}.pth"

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frozen = (args.mode == "probe")
    lr = 1e-3 if frozen else 1e-5

    train_dir = os.path.join(args.root, "train")
    test_dir  = os.path.join(args.root, "test")
    meta_tr   = read_metadata(os.path.join(args.root, "train_metadata.csv"))
    meta_te   = read_metadata(os.path.join(args.root, "test_metadata.csv"))

    tr_pids, va_pids = train_val_pids(meta_tr)
    print(f"Split: {len(tr_pids)} train / {len(va_pids)} val / {len(meta_te)} test "
          f"| task={args.task} mode={args.mode} lr={lr}")

    tr = DataLoader(ACDCDataset(train_dir, meta_tr, tr_pids, args.task, augment=True),
                    batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)
    va = DataLoader(ACDCDataset(train_dir, meta_tr, va_pids, args.task, augment=False),
                    batch_size=args.batch, shuffle=False, num_workers=4)
    te = DataLoader(ACDCDataset(test_dir, meta_te, list(meta_te.keys()), args.task, augment=False),
                    batch_size=args.batch, shuffle=False, num_workers=4)

    bot = load_pretrained_bot()
    if args.task == "clf":
        model = UxLSTMClassifier(bot, num_classes=len(CLASSES), freeze_encoder=frozen).to(device)
        criterion = nn.CrossEntropyLoss()
        better = lambda new, best: new > best
        key, best = "bal_acc", -1.0
    else:
        model = UxLSTMRegressor(bot, freeze_encoder=frozen).to(device)
        criterion = nn.SmoothL1Loss()
        better = lambda new, best: new < best
        key, best = "mae", 1e9

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for ep in range(1, args.epochs + 1):
        model.train()
        if frozen:
            model.backbone.eval()           # keep frozen InstanceNorm stats fixed
        running = 0.0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        sched.step()
        m = evaluate(model, va, args.task, device)
        flag = ""
        if better(m[key], best):
            best = m[key]
            torch.save({"model": model.state_dict(), "epoch": ep, "val": m,
                        "task": args.task, "mode": args.mode}, out)
            flag = "  <-- best (saved)"
        print(f"ep {ep:3d} | train_loss {running/len(tr.dataset):.4f} | "
              f"val {' '.join(f'{k}={v:.4f}' for k,v in m.items())}{flag}")

    # ── final test eval with best checkpoint ──────────────────────────────────────
    ckpt = torch.load(out, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_m = evaluate(model, te, args.task, device)
    print(f"\nBest val {key}={best:.4f} (ep {ckpt['epoch']})")
    print(f"TEST (50 held-out): {' '.join(f'{k}={v:.4f}' for k,v in test_m.items())}")
    print(f"Checkpoint -> {out}")


if __name__ == "__main__":
    main()
