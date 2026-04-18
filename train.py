import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import torchmetrics
import wandb

from config import CFG
from dataset import (
    PlantDiseaseDataset,
    build_disease_mapping,
    build_label_maps,
    make_weighted_sampler,
)
from model import build_model, freeze_backbone, get_optimizer, unfreeze_backbone
from transforms import get_train_transforms, get_val_transforms
from utils import plot_confusion_matrix, seed_everything


seed_everything(CFG.SEED)

FOLDER_TO_DISEASE = build_disease_mapping([CFG.TRAIN_DIR, CFG.VAL_DIR])
DISEASE2IDX, IDX2DISEASE = build_label_maps(FOLDER_TO_DISEASE)
NUM_CLASSES = len(DISEASE2IDX)
print(f"Unique disease classes: {NUM_CLASSES}")

train_ds = PlantDiseaseDataset(CFG.TRAIN_DIR, get_train_transforms(CFG.IMG_SIZE), FOLDER_TO_DISEASE, DISEASE2IDX)
val_ds   = PlantDiseaseDataset(CFG.VAL_DIR,   get_val_transforms(CFG.IMG_SIZE),   FOLDER_TO_DISEASE, DISEASE2IDX)

train_loader = DataLoader(
    train_ds,
    batch_size=CFG.BATCH_SIZE,
    sampler=make_weighted_sampler(train_ds),
    num_workers=CFG.NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=CFG.BATCH_SIZE * 2,
    shuffle=False,
    num_workers=CFG.NUM_WORKERS,
    pin_memory=True,
)

model     = build_model(CFG.BACKBONE, NUM_CLASSES, CFG.DROPOUT, CFG.DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=CFG.LABEL_SMOOTHING)
scaler    = GradScaler("cuda", enabled=CFG.AMP)

acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(CFG.DEVICE)
map_metric = torchmetrics.AveragePrecision(task="multiclass", num_classes=NUM_CLASSES, average="macro").to(CFG.DEVICE)

def mixup_data(x, y, alpha=0.4):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx   = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_one_epoch(epoch: int):
    model.train()
    total_loss = 0.0
    acc_metric.reset()

    for step, (imgs, labels) in enumerate(train_loader):
        imgs   = imgs.to(CFG.DEVICE, non_blocking=True)
        labels = labels.to(CFG.DEVICE, non_blocking=True)

        if CFG.USE_MIXUP:
            imgs, la, lb, lam = mixup_data(imgs, labels, CFG.MIXUP_ALPHA)

        with autocast("cuda", enabled=CFG.AMP):
            logits = model(imgs)
            loss   = mixup_criterion(logits, la, lb, lam) if CFG.USE_MIXUP else criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        acc_metric.update(logits.argmax(1), la if CFG.USE_MIXUP else labels)

        if step % 50 == 0:
            print(f"  Ep {epoch} | {step}/{len(train_loader)} | loss={loss.item():.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")

    return total_loss / len(train_loader), acc_metric.compute().item()

@torch.no_grad()
def validate(epoch: int):
    model.eval()
    total_loss = 0.0
    acc_metric.reset()
    map_metric.reset()
    all_preds, all_labels = [], []

    for imgs, labels in val_loader:
        imgs   = imgs.to(CFG.DEVICE, non_blocking=True)
        labels = labels.to(CFG.DEVICE, non_blocking=True)
        with autocast("cuda", enabled=CFG.AMP):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        probs = logits.softmax(1)
        total_loss += loss.item()
        acc_metric.update(probs.argmax(1), labels)
        map_metric.update(probs, labels)
        all_preds.extend(probs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (
        total_loss / len(val_loader),
        acc_metric.compute().item(),
        map_metric.compute().item(),
        all_preds,
        all_labels,
    )

def main(resume: str | None = None):
    global optimizer, scheduler

    start_epoch = 1
    best_map    = 0.0
    no_improve  = 0
    best_ckpt   = CFG.OUTPUT_DIR / "best_model.pth"

    if resume:
        print(f"Resuming from {resume}")
        ckpt = torch.load(resume, map_location=CFG.DEVICE)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt["epoch"] + 1
        best_map    = ckpt.get("val_map", 0.0)
        unfreeze_backbone(model)

    if CFG.USE_WANDB:
        wandb.init(project=CFG.WANDB_PROJECT, config=CFG.__dict__, resume="allow" if resume else None)

    for epoch in range(start_epoch, CFG.NUM_EPOCHS + 1):
        if epoch == 1:
            freeze_backbone(model)
            optimizer = get_optimizer(model, CFG.LR, CFG.WEIGHT_DECAY, unfrozen=False)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3, eta_min=1e-6)
            print("Phase 1 — head-only warm-up (backbone frozen)")

        if epoch == 4:
            unfreeze_backbone(model)
            optimizer = get_optimizer(model, CFG.LR, CFG.WEIGHT_DECAY, unfrozen=True)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=CFG.NUM_EPOCHS - 3, eta_min=1e-6
            )
            print("Phase 2 — full fine-tuning (backbone unfrozen)")

        print(f"\n{'='*55}\nEPOCH {epoch}/{CFG.NUM_EPOCHS}")
        tr_loss, tr_acc            = train_one_epoch(epoch)
        val_loss, val_acc, val_map, preds, labels = validate(epoch)
        scheduler.step()

        print(f"  TRAIN loss={tr_loss:.4f}  acc={tr_acc:.4f}")
        print(f"  VAL   loss={val_loss:.4f}  acc={val_acc:.4f}  mAP={val_map:.4f}")

        if CFG.USE_WANDB:
            wandb.log({"epoch": epoch, "train/loss": tr_loss, "train/acc": tr_acc,
                       "val/loss": val_loss, "val/acc": val_acc, "val/mAP": val_map,
                       "lr": optimizer.param_groups[0]["lr"]})

        if val_map > best_map:
            best_map   = val_map
            no_improve = 0
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "val_acc": val_acc, "val_map": val_map,
                "disease2idx": DISEASE2IDX, "backbone": CFG.BACKBONE,
            }, best_ckpt)
            print(f"  [SAVED] best mAP={best_map:.4f}")
        else:
            no_improve += 1
            if no_improve >= CFG.PATIENCE:
                print(f"  [EARLY STOP] patience={CFG.PATIENCE} reached.")
                break

    print("\nGenerating confusion matrix on best checkpoint …")
    ckpt = torch.load(best_ckpt, map_location=CFG.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    _, _, _, preds, labels = validate(epoch=-1)
    class_names = [IDX2DISEASE[i] for i in range(NUM_CLASSES)]
    plot_confusion_matrix(labels, preds, class_names, CFG.OUTPUT_DIR / "confusion_matrix.png")

    if CFG.USE_WANDB:
        wandb.finish()

    print(f"\nDone. Best val mAP: {best_map:.4f}  |  checkpoint: {best_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    main(resume=args.resume)
