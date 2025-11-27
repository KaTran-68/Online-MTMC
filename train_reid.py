#!/usr/bin/env python3
"""
Fine-tune re-id classifier (classification on IDs) using net_id_classifier.
Supports resuming from a saved checkpoint (weights + optimizer + scheduler + epoch + best_val).
"""
import os
import argparse
from glob import glob
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import random
from network import net_id_classifier
from collections import Counter

class ReIDFolderDataset(Dataset):
    def __init__(self, root, transform=None, labels_file=None):
        """
        root: path to train/ or val/ (folder with pid subfolders)
        labels_file: optional path to global labels.txt (pid label_index)
        If labels_file provided, we use that mapping (so train and val share same indices).
        Otherwise fallback to building mapping from folders in `root`.
        """
        self.samples = []
        self.transform = transform
        self.pid2label = {}

        # If labels_file provided and exists, load mapping
        if labels_file and os.path.isfile(labels_file):
            with open(labels_file, 'r') as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        pid = parts[0]
                        lbl = int(parts[1])
                        self.pid2label[pid] = lbl
            # collect only pids present on disk and present in mapping
            pids_on_disk = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            for pid in sorted(pids_on_disk):
                if pid not in self.pid2label:
                    # skip pid not in global mapping
                    continue
                imgs = sorted(glob(os.path.join(root, pid, '*.jpg')))
                for img in imgs:
                    self.samples.append((img, self.pid2label[pid]))
            if len(self.samples) == 0:
                raise RuntimeError(f"No images found in {root} matching labels file {labels_file}")
        else:
            # fallback: build mapping from folders in root
            pids = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
            self.pid2label = {pid:i for i,pid in enumerate(pids)}
            for pid in pids:
                imgs = sorted(glob(os.path.join(root, pid, '*.jpg')))
                for img in imgs:
                    self.samples.append((img, self.pid2label[pid]))
            if len(self.samples) == 0:
                raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, l = self.samples[idx]
        img = Image.open(p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, l

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='ReID root: datasets_reid')
    p.add_argument('--train-sub', default='train', help='train subfolder name (default train)')
    p.add_argument('--val-sub', default='val', help='val subfolder name (default val)')
    p.add_argument('--labels-file', default='', help='Optional labels file (pid label_index). Defaults to <data>/labels.txt if exists.')
    p.add_argument('--pretrained', default='', help='Path to checkpoint to init from (optional). Use with --resume to resume optimizer/scheduler/epoch.')
    p.add_argument('--resume', action='store_true', help='If set and --pretrained provided, resume optimizer/scheduler and epoch from checkpoint.')
    p.add_argument('--out', default='models/reid_finetune.pth.tar', help='Output checkpoint path')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.1, help='Initial learning rate (default 0.1 for SGD)')
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--workers', type=int, default=0, help='Number of dataloader workers (default 0 safe for Windows)')
    p.add_argument('--freeze-backbone-epochs', type=int, default=0, help='Freeze backbone first N epochs (default 0)')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

def build_model(num_classes, size_fc, arch='ResNet50'):
    model = net_id_classifier.net_id_classifier(arch, num_classes, size_fc)
    return model

def load_checkpoint_full(path, device):
    """
    Load a full checkpoint saved by this script.
    Returns a dict with keys possibly: state_dict, optimizer, scheduler, epoch, best_val
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(path)
    ck = torch.load(path, map_location=device)
    # ck expected to be dict with keys above; if older checkpoint only has state_dict, handle gracefully
    return ck

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = 0.0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        feats = model(imgs)  # returns embedding (batch, size_fc)
        logits = model.fc(feats)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item() * imgs.size(0)
        total += imgs.size(0)
    return running / total

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            feats = model(imgs)
            logits = model.fc(feats)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return correct / total if total>0 else 0.0

def load_labels_file(labels_file):
    pid2label = {}
    if labels_file and os.path.isfile(labels_file):
        with open(labels_file, 'r') as lf:
            for line in lf:
                parts = line.strip().split()
                if len(parts) >= 2:
                    pid2label[parts[0]] = int(parts[1])
    return pid2label

def dataset_stats(ds):
    counts = Counter([lbl for _,lbl in ds.samples])
    vals = list(counts.values())
    if not vals:
        return {}
    return {'num_ids': len(counts), 'num_imgs': sum(vals), 'min': min(vals), 'median': sorted(vals)[len(vals)//2], 'max': max(vals)}

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_root = os.path.join(args.data, args.train_sub)
    val_root = os.path.join(args.data, args.val_sub)
    labels_file = args.labels_file if args.labels_file else os.path.join(args.data, 'labels.txt')

    if not os.path.isdir(train_root):
        raise SystemExit(f"Train folder not found: {train_root}")

    transform_train = T.Compose([
        T.Resize((int(args.img_size*1.1), int(args.img_size*1.1))),
        T.RandomResizedCrop(args.img_size),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.1,0.1,0.1,0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    transform_val = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # Determine num_classes from labels file if present
    global_pid2label = load_labels_file(labels_file)
    if global_pid2label:
        num_classes = max(global_pid2label.values()) + 1
        print(f"Using labels file {labels_file} -> num_classes = {num_classes}")
    else:
        # fallback: will infer from train folders after dataset creation
        num_classes = None
        print("No labels file found or empty. Will infer labels from train folder.")

    # create datasets using (optional) labels file so train/val share same indices
    train_ds = ReIDFolderDataset(train_root, transform=transform_train, labels_file=labels_file if global_pid2label else None)
    if num_classes is None:
        num_classes = len({lbl for _,lbl in train_ds.samples})
    print(f"Num classes (from train): {num_classes}")
    train_stats = dataset_stats(train_ds)
    print("Train stats:", train_stats)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)

    val_loader = None
    if os.path.isdir(val_root):
        val_ds = ReIDFolderDataset(val_root, transform=transform_val, labels_file=labels_file if global_pid2label else None)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)
        print(f"Validation found with {len(val_ds)} images.")
        print("Val stats:", dataset_stats(val_ds))
    else:
        print("No validation folder found, skipping validation.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # size_fc default from config (2048)
    size_fc = 2048
    model = build_model(num_classes, size_fc, arch='ResNet50')

    # move model to device first (helpful if loading strict state dict that matches device)
    model = model.to(device)

    start_epoch = 1
    best_val = 0.0

    # If pretrained provided and resume requested, load checkpoint and restore optimizer/scheduler/epoch
    if args.pretrained and args.resume:
        try:
            ck = load_checkpoint_full(args.pretrained, device)
            # load model weights (state_dict may be nested)
            if isinstance(ck, dict) and 'state_dict' in ck:
                model.load_state_dict(ck['state_dict'], strict=False)
            elif isinstance(ck, dict):
                # older-style checkpoint may store full model keys directly
                # try load state dict if present, otherwise ignore
                if 'model_state_dict' in ck:
                    model.load_state_dict(ck['model_state_dict'], strict=False)
                elif any(k.startswith('module.') for k in ck.keys()):
                    # assume ck itself is state_dict
                    model.load_state_dict(ck, strict=False)
            else:
                print("Checkpoint format not recognized for model weights; skipping model load.")
            start_epoch = ck.get('epoch', 0) + 1 if isinstance(ck, dict) else 1
            best_val = ck.get('best_val', 0.0) if isinstance(ck, dict) else 0.0
            print(f"Resuming from checkpoint {args.pretrained}, start_epoch={start_epoch}, best_val={best_val}")
        except Exception as e:
            print("Warning: failed to load checkpoint for resume:", e)
            print("Proceeding without resume.")
            start_epoch = 1
            best_val = 0.0
    else:
        # If pretrained provided but not resume, load weights only (no optimizer/scheduler/epoch)
        if args.pretrained:
            try:
                ck = load_checkpoint_full(args.pretrained, device)
                if isinstance(ck, dict) and 'state_dict' in ck:
                    model.load_state_dict(ck['state_dict'], strict=False)
                elif isinstance(ck, dict) and 'model_state_dict' in ck:
                    model.load_state_dict(ck['model_state_dict'], strict=False)
                elif isinstance(ck, dict):
                    # maybe a plain state dict
                    try:
                        model.load_state_dict(ck, strict=False)
                    except Exception:
                        print("Could not load pretrained weights strictly; continuing with partial load.")
                else:
                    print("Pretrained format not recognized; skipping.")
                print(f"Loaded pretrained weights from {args.pretrained}")
            except Exception as e:
                print("Warning loading pretrained:", e)

    # Optionally freeze backbone parameters (all except model.fc) for first N epochs
    backbone_params = [p for n,p in model.named_parameters() if not n.startswith('fc')]
    fc_params = model.fc.parameters()

    criterion = nn.CrossEntropyLoss()

    # Use SGD as in paper with momentum and weight decay
    optimizer = torch.optim.SGD([
        {'params': backbone_params, 'lr': args.lr},
        {'params': fc_params, 'lr': args.lr}
    ], momentum=0.9, weight_decay=1e-4)

    # LR scheduler: step down by 0.1 every 25 epochs (matches paper)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    # If resuming and checkpoint contains optimizer/scheduler states, try to restore them
    if args.pretrained and args.resume:
        try:
            if 'optimizer' in ck and isinstance(ck['optimizer'], dict):
                optimizer.load_state_dict(ck['optimizer'])
                print("Optimizer state restored from checkpoint.")
            if 'scheduler' in ck and isinstance(ck['scheduler'], dict):
                # many schedulers store internal tensors on device; StepLR state_dict should be fine
                scheduler.load_state_dict(ck['scheduler'])
                print("Scheduler state restored from checkpoint.")
        except Exception as e:
            print("Warning: could not restore optimizer/scheduler state:", e)

    num_epochs = args.epochs
    best_val_local = best_val

    for epoch in range(start_epoch, num_epochs + 1):
        # freeze backbone if requested
        if args.freeze_backbone_epochs > 0 and epoch <= args.freeze_backbone_epochs:
            for p in backbone_params:
                p.requires_grad = False
            print(f"Epoch {epoch}: backbone frozen.")
        else:
            for p in backbone_params:
                p.requires_grad = True

        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch} train loss: {loss:.4f}")

        # step scheduler at epoch end (after training and validation)
        scheduler.step()
        # print current lr
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} lr: {current_lr:.6f}")

        # validation & saving
        if val_loader is not None:
            acc = evaluate(model, val_loader, device)
            print(f"Epoch {epoch} val acc: {acc*100:.2f}%")
            # Save checkpoint (full) every time we improve best_val OR every epoch (choose behavior)
            save_ckpt = False
            if acc > best_val_local:
                best_val_local = acc
                save_ckpt = True
                print(f"New best val acc: {best_val_local:.4f}")
            # always save epoch checkpoint (you can toggle this)
            save_epoch_ckpt = True

            if save_ckpt or save_epoch_ckpt:
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_val': best_val_local
                }
                # to avoid partial writes, save to temp then rename
                tmp_path = args.out + f".tmp_epoch{epoch}"
                torch.save(checkpoint, tmp_path)
                os.replace(tmp_path, args.out)
                print(f"Saved checkpoint to {args.out} (epoch {epoch}, val acc {acc:.4f})")
        else:
            # no val loader: save epoch checkpoint so you can resume later
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_val': best_val_local
            }
            tmp_path = args.out + f".tmp_epoch{epoch}"
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, args.out)
            print(f"Saved checkpoint to {args.out} (epoch {epoch})")

    print("Training complete. Best val acc:", best_val_local)

if __name__ == '__main__':
    main()