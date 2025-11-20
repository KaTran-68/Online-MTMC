import argparse, os, cv2
from collections import defaultdict

#  python visualize_tracks.py --video datasets\AIC19\test\S01\c001\c001.avi --tracks results\S01\prueba.txt --camera c001 --out results\S01\c001_viz.mp4
# python visualize_tracks.py --scene S01 --camera c001 --results results\S01\prueba.txt --dataset-root datasets\AIC19 --out results\S01\c001_viz.mp4

def parse_args():
    ap = argparse.ArgumentParser("Visualize MTMC results (format: id_cam id frame x y w h -1 -1)")
    ap.add_argument("--scene", required=True, help="e.g. S01")
    ap.add_argument("--camera", required=True, help="e.g. c001")
    ap.add_argument("--results", required=True, help="Path to results file, e.g. results\\S01\\prueba.txt")
    ap.add_argument("--dataset-root", required=True,
                    help="Root to dataset that contains 'test/<scene>/<camera>/*', "
                         "e.g. datasets\\AIC19 or datasets\\AIC19\\aic19-track1-mtmc")
    ap.add_argument("--out", required=True, help="Output video path, e.g. results\\S01\\c001_viz.mp4")
    ap.add_argument("--offset", type=float, default=0.0, help="Time offset in seconds for synchronization (default 0).")
    ap.add_argument("--score-th", type=float, default=-1.0, help="Ignored here (for compatibility).")
    ap.add_argument("--resize", type=float, default=1.0, help="Resize factor for output video.")
    ap.add_argument("--font-scale", type=float, default=0.8)
    ap.add_argument("--thickness", type=int, default=2)
    return ap.parse_args()

def color_for_id(tid: int):
    import random
    rnd = random.Random(tid)
    return (rnd.randint(64, 255), rnd.randint(64, 255), rnd.randint(64, 255))

def clamp_box(x, y, w, h, W, H):
    # clip bbox into frame; avoid negatives
    x = max(0, int(round(x)))
    y = max(0, int(round(y)))
    w = max(0, int(round(w)))
    h = max(0, int(round(h)))
    if x + w > W: w = max(0, W - x)
    if y + h > H: h = max(0, H - y)
    return x, y, w, h

def load_tracks_mtmc(path, cam_num):
    """
    Parse file lines of format:
    id_cam id frame x y w h -1 -1
    Keep only lines where id_cam == cam_num.
    Returns: dict frame_idx -> list[(tid, x, y, w, h)]
    """
    per_frame = defaultdict(list)
    bad, kept = 0, 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.replace(",", " ").split()
            if len(parts) < 7:
                bad += 1
                continue
            try:
                id_cam = int(parts[0])
                tid    = int(parts[1])
                frame  = int(parts[2])
                x      = float(parts[3])
                y      = float(parts[4])
                w      = float(parts[5])
                h      = float(parts[6])
            except Exception:
                bad += 1
                continue
            if id_cam != cam_num:
                continue
            per_frame[frame].append((tid, x, y, w, h))
            kept += 1
    if kept == 0:
        raise SystemExit(f"No tracks for camera id {cam_num} in {path}.")
    print(f"Loaded {kept} boxes for camera {cam_num} from {path}. Ignored {bad} malformed lines.")
    return per_frame

def find_video_path(root, scene, camera):
    cam_dir = os.path.join(root, "test", scene, camera)
    # try common video names
    for name in [f"{camera}.avi", "vdo.avi"]:
        p = os.path.join(cam_dir, name)
        if os.path.isfile(p):
            return p
    # fallback: first .avi
    for fn in os.listdir(cam_dir):
        if fn.lower().endswith(".avi"):
            return os.path.join(cam_dir, fn)
    return None

def main():
    args = parse_args()
    cam_num = int(args.camera[-3:])

    tracks = load_tracks_mtmc(args.results, cam_num)

    vid_path = find_video_path(args.dataset_root, args.scene, args.camera)
    if vid_path is None:
        raise SystemExit(f"Cannot find video under test/{args.scene}/{args.camera}.")

    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {vid_path}.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") if args.out.lower().endswith(".mp4") else cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (W, H))

    # Apply `offset` (convert seconds to frame)
    offset_frames = int(round(args.offset * fps))  # frame_shift based on offset seconds

    frame_idx, written = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_idx += 1

        for (tid, x, y, w, h) in tracks.get(frame_idx - offset_frames, []):
            pt1 = (int(round(x)), int(round(y)))
            pt2 = (int(round(x + w)), int(round(y + h)))
            color = color_for_id(tid)
            cv2.rectangle(frame, pt1, pt2, color, args.thickness)
            cv2.putText(frame, f"ID {tid}", (pt1[0], max(0, pt1[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, args.font_scale, color, max(1, args.thickness-1), cv2.LINE_AA)
        cv2.putText(frame, f"{args.scene} {args.camera} frame {frame_idx}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        written += 1

    writer.release()
    cap.release()

    print(f"Saved visualization: {args.out} (frames written: {written})")

if __name__ == "__main__":
    main()