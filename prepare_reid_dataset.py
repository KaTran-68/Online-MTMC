#!/usr/bin/env python3
"""
Prepare ReID dataset (classification) from AIC frames + GT (MOT-like).

Behavior:
- If --set train:
    - If --val-ratio > 0: split each PID's samples into train/val and save into out/train and out/val.
    - If --val-ratio == 0: save all samples into out/train.
- If --set validation:
    - Save all samples into out/val (no splitting).
- Finally, write out/labels.txt mapping pid -> label_index based on pids present in out/train and out/val.
"""
import os
import argparse
import random
from collections import defaultdict
from PIL import Image

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', required=True, help='Root dataset, e.g. ./datasets/AIC19')
    p.add_argument('--set', default='train', choices=['train', 'validation', 'test'],
                   help='Which set to process: train (default) or validation/test')
    p.add_argument('--out', default='datasets_reid', help='Output root for reid dataset')
    p.add_argument('--val-ratio', type=float, default=0.1, help='Fraction for validation split (0 to disable). Only used with --set train')
    p.add_argument('--min-area', type=int, default=500, help='Minimum bbox area to keep (w*h)')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

def clamp(bbox, W, H):
    x, y, w, h = bbox
    x = int(round(x)); y = int(round(y)); w = int(round(w)); h = int(round(h))
    if w <= 0 or h <= 0:
        return None
    if x < 0:
        w = w + x
        x = 0
    if y < 0:
        h = h + y
        y = 0
    if x + w > W:
        w = W - x
    if y + h > H:
        h = H - y
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)

def find_gt_file(gt_dir):
    # common names: gt, gt.txt, S01.txt, etc. Return first txt file or 'gt'
    cand = ['gt', 'gt.txt']
    for name in cand:
        p = os.path.join(gt_dir, name)
        if os.path.isfile(p):
            return p
    # fallback: any .txt file
    for fn in os.listdir(gt_dir):
        if fn.lower().endswith('.txt'):
            return os.path.join(gt_dir, fn)
    return None

def collect_crops_from_set(dataset_root, set_name):
    crops = defaultdict(list)  # pid -> list of (frame_path, bbox)
    scenes_dir = os.path.join(dataset_root, set_name)
    if not os.path.isdir(scenes_dir):
        raise SystemExit(f"Cannot find {scenes_dir}")
    scenes = sorted([d for d in os.listdir(scenes_dir) if os.path.isdir(os.path.join(scenes_dir, d))])
    print(f"Found scenes in {set_name}: {scenes}")

    for s in scenes:
        scene_dir = os.path.join(scenes_dir, s)
        cameras = sorted([d for d in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, d))])
        for c in cameras:
            cam_dir = os.path.join(scene_dir, c)
            img_dir = os.path.join(cam_dir, 'img')
            gt_dir = os.path.join(cam_dir, 'gt')
            if not os.path.isdir(img_dir) or not os.path.isdir(gt_dir):
                print(f"Skip {s}/{c} - missing img or gt")
                continue
            gt_file = find_gt_file(gt_dir)
            if gt_file is None:
                print(f"No gt file found in {gt_dir}, skipping")
                continue
            print(f"Processing GT {gt_file} for {s}/{c}")

            with open(gt_file, 'r') as gf:
                for line in gf:
                    line = line.strip()
                    if not line:
                        continue
                    parts = [p.strip() for p in line.replace(',', ' ').split()]
                    if len(parts) < 6:
                        continue
                    try:
                        frame = int(float(parts[0]))
                        pid = parts[1]
                        x = float(parts[2]); y = float(parts[3]); w = float(parts[4]); h = float(parts[5])
                    except Exception:
                        continue
                    frame_name = os.path.join(img_dir, f"{frame:06d}.jpg")
                    if not os.path.isfile(frame_name):
                        # try without zero pad
                        frame_name_alt = os.path.join(img_dir, f"{frame}.jpg")
                        if os.path.isfile(frame_name_alt):
                            frame_name = frame_name_alt
                        else:
                            continue
                    crops[pid].append((frame_name, (x, y, w, h)))
    return crops

def save_items_to_folder(items, out_folder, pid, min_area):
    os.makedirs(out_folder, exist_ok=True)
    cnt = 0
    saved = 0
    for frame_path, bbox in items:
        try:
            im = Image.open(frame_path).convert('RGB')
        except Exception:
            continue
        W, H = im.size[0], im.size[1]
        bb = clamp(bbox, W, H)
        if bb is None:
            continue
        x, y, w, h = bb
        if w * h < min_area:
            continue
        crop = im.crop((x, y, x + w, y + h))
        fname = f"{pid}_{cnt:06d}.jpg"
        crop.save(os.path.join(out_folder, fname), quality=95)
        cnt += 1
        saved += 1
    return saved

def write_labels_from_out(out_root):
    # collect pids from out/train and out/val
    pids = set()
    train_dir = os.path.join(out_root, 'train')
    val_dir = os.path.join(out_root, 'val')
    if os.path.isdir(train_dir):
        for d in os.listdir(train_dir):
            if os.path.isdir(os.path.join(train_dir, d)):
                pids.add(d)
    if os.path.isdir(val_dir):
        for d in os.listdir(val_dir):
            if os.path.isdir(os.path.join(val_dir, d)):
                pids.add(d)
    pids = sorted(pids)
    labels_file = os.path.join(out_root, 'labels.txt')
    with open(labels_file, 'w') as lf:
        for i, pid in enumerate(pids):
            lf.write(f"{pid} {i}\n")
    return len(pids), labels_file

def main():
    args = parse_args()
    random.seed(args.seed)

    dataset_root = args.dataset_root
    set_name = args.set
    out_root = args.out
    val_ratio = args.val_ratio
    min_area = args.min_area

    # collect crops from the requested set
    crops = collect_crops_from_set(dataset_root, set_name)

    out_train = os.path.join(out_root, 'train')
    out_val = os.path.join(out_root, 'val')
    os.makedirs(out_root, exist_ok=True)

    total_saved = 0
    # If set == 'validation' -> save everything to out/val
    if set_name == 'validation':
        print(f"Saving validation crops into {out_val}")
        for pid, items in sorted(crops.items(), key=lambda x: x[0]):
            if len(items) == 0:
                continue
            pid_val_folder = os.path.join(out_val, pid)
            saved = save_items_to_folder(items, pid_val_folder, pid, min_area)
            total_saved += saved
    else:
        # set == 'train' or others: either split or save all to train
        print(f"Saving train crops into {out_train}")
        for pid, items in sorted(crops.items(), key=lambda x: x[0]):
            if len(items) == 0:
                continue
            random.shuffle(items)
            if val_ratio > 0:
                n_val = int(len(items) * val_ratio)
                val_items = items[:n_val]
                train_items = items[n_val:]
            else:
                val_items = []
                train_items = items

            # save train items
            pid_train_folder = os.path.join(out_train, pid)
            saved_train = save_items_to_folder(train_items, pid_train_folder, pid, min_area)
            total_saved += saved_train

            # save val items (if any)
            if val_items:
                pid_val_folder = os.path.join(out_val, pid)
                saved_val = save_items_to_folder(val_items, pid_val_folder, pid, min_area)
                total_saved += saved_val

    # Write labels mapping based on folders present in out_root (train and val)
    num_pids, labels_file = write_labels_from_out(out_root)

    print(f"Saved {total_saved} crops for {num_pids} IDs into {out_root}")
    print(f"Labels mapping written to {labels_file}")
    print("Done.")

if __name__ == '__main__':
    main()