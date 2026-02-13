"""
train_resnet18_originalCrops_plus_ablated_across_alphas__SAME_SETTINGS_AS_BEFORE.py

This uses the SAME exact settings as your earlier 30-trial ablation experiment:
- TRIALS = 30
- EPOCHS = 50
- LR = 1e-4
- IMG_SIZE = 224
- BATCH_SIZE = 32
- NUM_WORKERS = 0
- Train transforms: Resize(224), RandomHFlip, RandomRotation(10), ToTensor, Normalize([0.5],[0.5])
- Eval transforms:  Resize(224), ToTensor, Normalize([0.5],[0.5])
- Model: ResNet18 pretrained on ImageNet
    - 1-channel conv1 initialized by averaging pretrained RGB conv1 weights
    - Freeze all params, then unfreeze layer3, layer4 (and fc via replacement)
    - Binary head: fc -> Linear(in_features, 1)
- Loss: BCEWithLogitsLoss(pos_weight = n_neg / n_pos) computed per trial TRAIN split
- Split per trial: stratified 70/10/20 using:
    - train_test_split(df_final, test_size=0.30, stratify=label, random_state=seed)
    - train_test_split(df_temp, test_size=2/3, stratify=label, random_state=seed)
      (same as your prior code; uses the SAME seed for both splits)
- Threshold: sigmoid(logit) >= 0.5
- For each alpha:
    df_final = ORIGINAL_CROPS_BASE + SELECTED_ABLATED(alpha)
    SELECTED_ABLATED(alpha): rows where (LRR_Labels==0) & (nn_pred_label==1), label forced to 1

You MUST set:
- ORIGINAL_CROPS_CSV
- ORIGINAL_CROPS_ROOT
- MANIFEST_PATHS

Outputs:
- RESULTS_DIR/
    - alpha_<alpha_tag>/
        - model_alpha_<alpha>_trial_XX.pt
        - history_alpha_<alpha>_trial_XX.csv
        - metrics_alpha_<alpha>_trial_XX.json
    - all_trials_alpha_<alpha>.csv
    - summary_alpha_<alpha>.csv
    - all_alphas_summary.csv
"""

import os
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms as T

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# ============================================================
# 0) PATHS YOU MUST SET
# ============================================================

# Original crops CSV (your screenshot)
ORIGINAL_CROPS_CSV = Path(r"D:\hello\orignial_crops_file.csv")

# Directory that CONTAINS the folder referenced in image_path.
# Example: image_path = "lidc_orig_norm_images\\1.jpg"
# then ORIGINAL_CROPS_ROOT must be the folder that contains "lidc_orig_norm_images".
ORIGINAL_CROPS_ROOT = Path(r"D:\hello")

# Ablated/pixelated manifests per alpha (manifest_with_nn.csv)
MANIFEST_PATHS = [
    r"D:\hello\Model&AblationCode\images\ablated_false_noise\alpha_0.1\manifest_with_nn.csv",
    r"D:\hello\Model&AblationCode\images\ablated_false_noise\alpha_0.2\manifest_with_nn.csv",
    r"D:\hello\Model&AblationCode\images\ablated_false_noise\alpha_0.3\manifest_with_nn.csv",
    r"D:\hello\Model&AblationCode\images\ablated_false_noise\alpha_0.4\manifest_with_nn.csv",
    r"D:\hello\Model&AblationCode\images\ablated_false_noise\alpha_0.5\manifest_with_nn.csv",
    r"D:\hello\Model&AblationCode\images\ablated_false_noise\alpha_0.6\manifest_with_nn.csv",
    r"D:\hello\Model&AblationCode\images\ablated_false_noise\alpha_0.7\manifest_with_nn.csv",
    r"D:\hello\Model&AblationCode\images\ablated_false_noise\alpha_0.8\manifest_with_nn.csv",
    r"D:\hello\Model&AblationCode\images\ablated_false_noise\alpha_0.9\manifest_with_nn.csv",
    r"D:\hello\Model&AblationCode\images\ablated_false_noise\alpha_1\manifest_with_nn.csv",
    r"D:\hello\Model&AblationCode\images\ablated_false_noise\alpha_2\manifest_with_nn.csv",
    r"D:\hello\Model&AblationCode\images\ablated_false_noise\alpha_3\manifest_with_nn.csv",
]


# Output dir
RESULTS_DIR = Path("resnet18_originalCrops_plus_ablated__same_settings_30trials_50epochs")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1) EXACT SETTINGS (MATCHES YOUR PRIOR 30-TRIAL ABLATION CODE)
# ============================================================

TRIALS = 30
EPOCHS = 50
LR = 1e-4

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0  # same as before

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# ============================================================
# 2) COLUMN NAMES (adjust ONLY if your files differ)
# ============================================================

# Original crops csv columns
CROP_PATH_COL = "image_path"
CROP_SPIC_COL = "spiculated"
CROP_NONSPIC_COL = "nonSpiculated"  # not used for label; kept for sanity if needed

# Manifest columns
MAN_FILEPATH_COL = "filepath"
MAN_LRR_COL = "LRR_Labels"
MAN_NN_COL = "nn_pred_label"


# ============================================================
# 3) LOGGING (OPTIONAL) - same style as your prior run_log setup
# ============================================================
# Uncomment if you want file logging exactly like your earlier script:
#
# import sys
# LOG_FILE = "run_log_originalCrops_plus_ablated.txt"
# log_f = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
# sys.stdout = log_f
# sys.stderr = log_f
# print("\n==============================")
# print("Run started:", datetime.now())
# print("Python:", sys.executable)


# ============================================================
# 4) SEEDING (SAME INTENT AS BEFORE: trial-based; controls split)
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism flags (same approach as we discussed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 5) TRANSFORMS / DATASET (SAME AS BEFORE)
# ============================================================
train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])

eval_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])


class FilepathDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["filepath"]
        label = int(row["label"])

        img = Image.open(path).convert("L")
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


# ============================================================
# 6) MODEL / METRICS (SAME AS BEFORE, FIXED conv1 INIT)
# ============================================================
def make_resnet18(freeze_backbone=True, unfreeze_layers=("layer3", "layer4")):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # 1-channel input with pretrained init (RGB mean)
    if model.conv1.in_channels == 3:
        old_w = model.conv1.weight.data  # [64,3,7,7]
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv.weight.data = old_w.mean(dim=1, keepdim=True)
        model.conv1 = new_conv

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for layer_name in unfreeze_layers:
            layer = getattr(model, layer_name)
            for p in layer.parameters():
                p.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, 1)  # 1 logit
    return model.to(DEVICE)


def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc  = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    return {
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


@torch.no_grad()
def eval_loader(model, loader):
    model.eval()
    all_true, all_pred = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        logits = model(imgs).view(-1)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
        all_true.extend(labels.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())
    return compute_metrics(all_true, all_pred)


def train_resnet18(df_train, train_dl, val_dl, num_epochs=50, lr=1e-4, save_path=None):
    model = make_resnet18(freeze_backbone=True, unfreeze_layers=("layer3", "layer4"))

    labels_np = df_train["label"].to_numpy()
    n_pos = int((labels_np == 1).sum())
    n_neg = int((labels_np == 0).sum())
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-8)], device=DEVICE, dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_val_acc = -1.0
    best_state = None
    history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for imgs, labels in train_dl:
            imgs = imgs.to(DEVICE)
            labels = labels.float().to(DEVICE)

            optimizer.zero_grad()
            logits = model(imgs).view(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / max(1, len(train_dl.dataset))

        val_metrics = eval_loader(model, val_dl)
        val_acc = float(val_metrics["accuracy"])

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_sensitivity": float(val_metrics["sensitivity"]),
            "val_specificity": float(val_metrics["specificity"]),
            "val_tp": int(val_metrics["tp"]),
            "val_tn": int(val_metrics["tn"]),
            "val_fp": int(val_metrics["fp"]),
            "val_fn": int(val_metrics["fn"]),
            "pos_weight": float(pos_weight.item()),
            "n_pos_train": n_pos,
            "n_neg_train": n_neg,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            if save_path is not None:
                torch.save(best_state, str(save_path))

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, pd.DataFrame(history), float(best_val_acc)


# ============================================================
# 7) BUILD BASE FROM ORIGINAL CROPS CSV (REAL PATHS)
# ============================================================
def make_absolute_crop_path(rel_path: str) -> str:
    rel = Path(str(rel_path).replace("/", os.sep).replace("\\", os.sep))
    return str((ORIGINAL_CROPS_ROOT / rel).resolve())


def load_original_crops_base() -> pd.DataFrame:
    if not ORIGINAL_CROPS_CSV.exists():
        raise FileNotFoundError(f"Missing ORIGINAL_CROPS_CSV: {ORIGINAL_CROPS_CSV}")

    df = pd.read_csv(ORIGINAL_CROPS_CSV)
    need = [CROP_PATH_COL, CROP_SPIC_COL, CROP_NONSPIC_COL]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Original crops CSV missing columns: {missing}")

    # SAME label idea: label = spiculated (0/1)
    df["label"] = df[CROP_SPIC_COL].astype(int)

    # absolute file path to the REAL original crop image
    df["filepath"] = df[CROP_PATH_COL].apply(make_absolute_crop_path)

    out = df[["filepath", "label"]].copy()
    out["label"] = out["label"].astype(int)

    counts = out["label"].value_counts().to_dict()
    print("Original crops base label counts:", counts)

    # Sanity: missing files check
    missing_files = int((out["filepath"].apply(lambda p: not Path(p).exists())).sum())
    if missing_files > 0:
        print(f"⚠️ Missing original crop files: {missing_files}  -> check ORIGINAL_CROPS_ROOT")

    return out


# ============================================================
# 8) LOAD SELECTED ABLATED PER ALPHA (SAME SELECTION RULE)
# ============================================================
def alpha_tag_from_manifest_path(p: str) -> str:
    parent = Path(p).parent.name  # alpha_0.1
    if parent.startswith("alpha_"):
        return parent.replace("alpha_", "")
    return Path(p).stem


def load_selected_ablated(manifest_path: str, alpha_tag: str) -> pd.DataFrame:
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    m = pd.read_csv(p)
    need = [MAN_FILEPATH_COL, MAN_LRR_COL, MAN_NN_COL]
    missing = [c for c in need if c not in m.columns]
    if missing:
        raise KeyError(f"Manifest {manifest_path} missing columns: {missing}")

    sel = m[(m[MAN_LRR_COL] == 0) & (m[MAN_NN_COL] == 1)].copy()
    sel = sel[[MAN_FILEPATH_COL, MAN_NN_COL]].rename(columns={MAN_NN_COL: "label"}).copy()
    sel["label"] = sel["label"].astype(int)

    # SAME behavior as earlier: force label to 1 since you filtered nn_pred_label==1
    sel["label"] = 1

    return sel


# ============================================================
# 9) MAIN: PER-ALPHA LOOP + 30 TRIALS (SAME AS BEFORE)
# ============================================================
def main():
    print("Using device:", DEVICE)
    print("RESULTS_DIR:", RESULTS_DIR.resolve())

    df_original_base = load_original_crops_base()

    all_alpha_summary_rows = []

    for manifest_path in MANIFEST_PATHS:
        alpha_tag = alpha_tag_from_manifest_path(manifest_path)
        print(f"\n==================== alpha = {alpha_tag} ====================")

        df_ablated = load_selected_ablated(manifest_path, alpha_tag)
        print("Selected ablated samples:", len(df_ablated))
        if len(df_ablated) == 0:
            print("⚠️ No transformed/ablated samples found. Skipping alpha.")
            continue

        # Combine base + ablated(alpha)
        df_final = pd.concat([df_original_base, df_ablated], ignore_index=True)
        df_final["label"] = df_final["label"].astype(int)

        # Trial outputs for this alpha
        alpha_dir = RESULTS_DIR / f"alpha_{alpha_tag}"
        alpha_dir.mkdir(parents=True, exist_ok=True)

        trial_rows = []

        for trial in range(1, TRIALS + 1):
            seed = 1000 + trial  # SAME as your earlier 30-trials code
            set_seed(seed)

            print(f"\n--- Trial {trial}/{TRIALS} (seed={seed}) ---")

            # SAME split logic (and SAME random_state usage) as your earlier script
            df_train, df_temp = train_test_split(
                df_final, test_size=0.30, stratify=df_final["label"], random_state=seed
            )
            df_val, df_test = train_test_split(
                df_temp, test_size=2/3, stratify=df_temp["label"], random_state=seed
            )

            train_ds = FilepathDataset(df_train, transform=train_transform)
            val_ds = FilepathDataset(df_val, transform=eval_transform)
            test_ds = FilepathDataset(df_test, transform=eval_transform)

            train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
            val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

            model_path   = alpha_dir / f"model_alpha_{alpha_tag}_trial_{trial:02d}.pt"
            history_path = alpha_dir / f"history_alpha_{alpha_tag}_trial_{trial:02d}.csv"
            metrics_path = alpha_dir / f"metrics_alpha_{alpha_tag}_trial_{trial:02d}.json"

            model, history_df, best_val_acc = train_resnet18(
                df_train=df_train,
                train_dl=train_dl,
                val_dl=val_dl,
                num_epochs=EPOCHS,
                lr=LR,
                save_path=model_path
            )

            history_df.to_csv(history_path, index=False)

            val_metrics  = eval_loader(model, val_dl)
            test_metrics = eval_loader(model, test_dl)

            meta = {
                "alpha": alpha_tag,
                "trial": trial,
                "seed": seed,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "epochs": EPOCHS,
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "img_size": IMG_SIZE,
                "train_size": int(len(df_train)),
                "val_size": int(len(df_val)),
                "test_size": int(len(df_test)),
                "train_class_balance": {
                    "0": int((df_train["label"] == 0).sum()),
                    "1": int((df_train["label"] == 1).sum()),
                },
                "num_original_crops": int(len(df_original_base)),
                "num_selected_ablated": int(len(df_ablated)),
                "manifest_path": str(manifest_path),
                "model_path": str(model_path),
                "history_path": str(history_path),
                "best_val_acc_during_training": float(best_val_acc),
                "trainable_layers": ["layer3", "layer4", "fc"],
                "frozen_layers": ["conv1", "bn1", "layer1", "layer2"],
                "conv1_init": "pretrained_rgb_mean_to_1ch",
                "threshold": 0.5,
            }

            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump({"meta": meta, "val_metrics": val_metrics, "test_metrics": test_metrics}, f, indent=2)

            trial_rows.append({
                "alpha": alpha_tag,
                "trial": trial,
                "seed": seed,
                "val_accuracy": val_metrics["accuracy"],
                "val_sensitivity": val_metrics["sensitivity"],
                "val_specificity": val_metrics["specificity"],
                "test_accuracy": test_metrics["accuracy"],
                "test_sensitivity": test_metrics["sensitivity"],
                "test_specificity": test_metrics["specificity"],
            })

            print("VAL:", val_metrics, "| TEST:", test_metrics)

        # Save all trials for this alpha
        trials_df = pd.DataFrame(trial_rows)
        trials_csv = RESULTS_DIR / f"all_trials_alpha_{alpha_tag}.csv"
        trials_df.to_csv(trials_csv, index=False)

        # Save mean/std summary for this alpha
        metric_cols = [
            "val_accuracy", "val_sensitivity", "val_specificity",
            "test_accuracy", "test_sensitivity", "test_specificity"
        ]
        summary_df = trials_df[metric_cols].agg(["mean", "std"]).reset_index().rename(columns={"index": "stat"})
        summary_csv = RESULTS_DIR / f"summary_alpha_{alpha_tag}.csv"
        summary_df.to_csv(summary_csv, index=False)

        print(f"\n✅ Saved all trials: {trials_csv}")
        print(f"✅ Saved summary:   {summary_csv}")

        # Add to global summary across alphas
        means = trials_df[metric_cols].mean(numeric_only=True)
        stds  = trials_df[metric_cols].std(numeric_only=True)

        row = {"alpha": alpha_tag}
        for c in metric_cols:
            row[f"{c}_mean"] = float(means[c])
            row[f"{c}_std"]  = float(stds[c])

        all_alpha_summary_rows.append(row)

    # Save global summary
    if all_alpha_summary_rows:
        all_alpha_summary = pd.DataFrame(all_alpha_summary_rows)
        out_csv = RESULTS_DIR / "all_alphas_summary.csv"
        all_alpha_summary.to_csv(out_csv, index=False)
        print(f"\n✅ Saved global alpha summary: {out_csv}")
    else:
        print("\n⚠️ No alpha results were produced.")


if __name__ == "__main__":
    main()
