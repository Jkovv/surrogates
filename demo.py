#!/usr/bin/env python3

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

CYTOKINE_NAMES = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
MASK_IDS = [1, 2, 3, 4, 5]
VTK_FIELDS = ["CellType"] + CYTOKINE_NAMES
WINDOW = 2

REPO_ROOT = Path(__file__).resolve().parent
DEMO_2D = REPO_ROOT / "demo_data" / "2d"
DEMO_3D = REPO_ROOT / "demo_data" / "3d"


def list_frames(folder):
    folder = Path(folder)
    direct = sorted([folder / f for f in os.listdir(folder) if f.endswith(".vtk")],
                    key=lambda p: int("".join(filter(str.isdigit, p.name)) or 0))
    if direct:
        return direct
    best = []
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
        fs = sorted([sub / f for f in os.listdir(sub) if f.endswith(".vtk")],
                    key=lambda p: int("".join(filter(str.isdigit, p.name)) or 0))
        if len(fs) > len(best):
            best = fs
    return best


def read_frame(pv, path):
    m = pv.read(str(path))
    dx, dy, dz = m.dimensions
    is_2d = (dz == 1)
    shape = (dx, dy) if is_2d else (dx, dy, dz)
    cyto = np.stack([m.point_data[c].reshape(shape, order="F")
                     for c in CYTOKINE_NAMES], axis=-1).astype(np.float32)
    if "CellType" in m.point_data:
        ct = m.point_data["CellType"].reshape(shape, order="F")
    else:
        ct = np.zeros(shape, dtype=np.int16)
    return cyto, ct, is_2d


def load_trajectory(folder, corner, max_frames):
    import pyvista as pv
    frames = list_frames(folder)
    if not frames:
        sys.exit(f"ERROR: no Step_*.vtk frames under {folder}")
    if max_frames:
        frames = frames[:max_frames]
    c = corner
    cyto_list, mask_list, is_2d = [], [], None
    for fp in frames:
        cyto, ct, is_2d = read_frame(pv, fp)
        if is_2d:
            cyto_c, ct_c = cyto[:c, :c], ct[:c, :c]
            m = np.zeros((c, c, 5), np.float32)
        else:
            cyto_c, ct_c = cyto[:c, :c, :c], ct[:c, :c, :c]
            m = np.zeros((c, c, c, 5), np.float32)
        for ch, cid in enumerate(MASK_IDS):
            m[..., ch] = (ct_c == cid)
        cyto_list.append(cyto_c)
        mask_list.append(m)
    return np.stack(cyto_list), np.stack(mask_list), is_2d


def carve_demo(pv, src_folder, out_folder, corner, n_frames):
    frames = list_frames(src_folder)
    if not frames:
        sys.exit(f"ERROR: no Step_*.vtk frames under {src_folder}")
    frames = frames[:n_frames]
    out_folder.mkdir(parents=True, exist_ok=True)
    c = corner
    for fp in frames:
        m = pv.read(str(fp))
        dx, dy, dz = m.dimensions
        is_2d = (dz == 1)
        full_shape = (dx, dy) if is_2d else (dx, dy, dz)
        out_dims = (c, c, 1) if is_2d else (c, c, c)
        out = pv.ImageData(dimensions=out_dims)
        for name in VTK_FIELDS:
            if name not in m.point_data:
                continue
            full = m.point_data[name].reshape(full_shape, order="F")
            corner_arr = full[:c, :c] if is_2d else full[:c, :c, :c]
            out.point_data[name] = corner_arr.reshape(-1, order="F")
        out.save(str(out_folder / fp.name))
    print(f"[carve] {len(frames)} frames {full_shape} -> {out_dims} into {out_folder}")


NOISE_FLOOR_FRAC = 1e-4


def clip_pct(kappa):
    if kappa >= 600:
        return 98.0
    if kappa >= 300:
        return 98.5
    if kappa >= 100:
        return 99.0
    if kappa >= 20:
        return 99.5
    return None


def fit_scale(train_cyto):
    from scipy.stats import kurtosis
    flat = train_cyto.reshape(-1, 6).astype(np.float64)
    cmax_raw = np.maximum(flat.max(axis=0), 1e-300)
    floored = np.where(flat >= NOISE_FLOOR_FRAC * cmax_raw, flat, 0.0)
    cmax = np.zeros(6)
    for c in range(6):
        col = floored[:, c]
        kap = float(kurtosis(col, fisher=True, bias=False))
        q = clip_pct(kap)
        if q is None:
            cmax[c] = float(np.max(col))
        else:
            pv = float(np.percentile(col, q))
            cmax[c] = pv if pv > NOISE_FLOOR_FRAC * cmax_raw[c] else float(np.max(col))
    return np.maximum(cmax, 1e-12), cmax_raw


def apply_scale(cyto, cmax, cmax_raw):
    floored = np.where(cyto >= NOISE_FLOOR_FRAC * cmax_raw, cyto, 0.0)
    clipped = np.minimum(floored, cmax)
    return (2.0 * clipped / cmax - 1.0).astype(np.float32)


def preprocess(cyto_traj, mask_traj, out_dir, train_frac=0.7):
    T = cyto_traj.shape[0]
    if T < WINDOW + 1:
        sys.exit(f"ERROR: need >= {WINDOW + 1} frames, got {T}")
    spatial = cyto_traj.shape[1:-1]
    G = spatial[0]
    dim = len(spatial)
    n = T - WINDOW

    n_tr = max(1, int(n * train_frac))
    cmax, cmax_raw = fit_scale(cyto_traj[WINDOW:WINDOW + n_tr])
    scaled = apply_scale(cyto_traj, cmax, cmax_raw)

    Y_target = scaled[WINDOW:]
    Y_masks = mask_traj[WINDOW:]
    cyto_seq = np.stack([scaled[i:i + WINDOW] for i in range(n)])
    mask_seq = np.stack([mask_traj[i:i + WINDOW] for i in range(n)])
    X_branch = np.concatenate([cyto_seq, mask_seq], axis=-1)

    xs = np.linspace(-1.0, 1.0, G, dtype=np.float32)
    mesh = np.meshgrid(*([xs] * dim), indexing="ij")
    coords = np.stack([mm.ravel() for mm in mesh], axis=-1)
    t_norm = np.linspace(-1.0, 1.0, T, dtype=np.float32)[WINDOW:]
    P = coords.shape[0]
    X_trunk = np.zeros((n, P, dim + 1), dtype=np.float32)
    for i in range(n):
        X_trunk[i, :, :dim] = coords
        X_trunk[i, :, dim] = t_norm[i]

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_branch.npy", X_branch.astype(np.float32))
    np.save(out_dir / "X_trunk.npy", X_trunk)
    np.save(out_dir / "Y_target.npy", Y_target.astype(np.float32))
    np.save(out_dir / "Y_masks_spatial.npy", Y_masks.astype(np.float32))
    meta = {"grid": int(G), "dim": dim, "n_samples": int(n), "window": WINDOW,
            "cytokines": CYTOKINE_NAMES, "mask_ids": MASK_IDS,
            "scaling": {"kind": "kurtosis_adaptive_clip_-1_1",
                        "max": cmax.tolist(),
                        "denorm": "x_phys=(x+1)/2*max[j]"}}
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[prep]  {dim}D {'x'.join(str(s) for s in spatial)}: "
          f"n={n}  range[{scaled.min():+.3f},{scaled.max():+.3f}]")
    return out_dir


def fisher_z(r):
    return np.arctanh(min(max(r, -0.999999), 0.999999))


def calculate_metrics(y_true, y_pred, masks, clip_max):
    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr
    from skimage.metrics import structural_similarity as ssim
    T = min(y_true.shape[0], y_pred.shape[0], masks.shape[0])
    yt = y_true[:T]
    yp = np.maximum(y_pred[:T], 0.0)
    ms = np.max(masks[:T], axis=-1, keepdims=True)
    sq = np.square(yt - yp)
    rmse = float(np.sqrt(np.sum(sq * ms) / (np.sum(ms) + 1e-12)))
    unmasked = float(np.sqrt(np.mean(sq)))
    r2 = float(r2_score(yt.flatten(), yp.flatten()))
    per_t = []
    for t in range(T):
        g = yt[t].flatten()
        per_t.append(float(r2_score(g, yp[t].flatten())) if np.std(g) > 1e-12 else np.nan)
    thr = 0.05 * clip_max if clip_max > 0 else 1e-9
    dr = float(clip_max) if clip_max > 0 else 1.0
    dices, zc, ss = [], [], []
    n_empty = n_skip = 0
    for t in range(T):
        gt = yt[t, ..., 0]
        pr = yp[t, ..., 0]
        gb = (gt > thr).astype(float)
        pb = (pr > thr).astype(float)
        if gb.sum() + pb.sum() == 0:
            n_empty += 1
        else:
            dices.append(2 * np.sum(gb * pb) / (gb.sum() + pb.sum() + 1e-12))
        if np.std(gt) > 1e-12 and np.std(pr) > 1e-12:
            rv = float(pearsonr(gt.flatten(), pr.flatten())[0])
            if np.isfinite(rv):
                zc.append(fisher_z(rv))
        if float(np.max(gt) - np.min(gt)) > 1e-12:
            win = min(7, gt.shape[0])
            if win % 2 == 0:
                win -= 1
            try:
                ss.append(float(ssim(gt, pr, data_range=dr, win_size=max(3, win))))
            except Exception:
                n_skip += 1
        else:
            n_skip += 1
    return {"Global_R2": r2,
            "Per_Timestep_R2_mean": float(np.nanmean(per_t)) if per_t else 0.0,
            "Masked_RMSE": rmse, "Unmasked_RMSE": unmasked,
            "Avg_Dice": float(np.mean(dices)) if dices else 0.0,
            "Dice_Empty_Skipped": n_empty,
            "Spatial_Correlation": float(np.tanh(np.mean(zc))) if zc else 0.0,
            "SSIM": float(np.mean(ss)) if ss else 0.0,
            "SSIM_Skipped_Frames": n_skip}


def denormalize(x, clip_max):
    return (np.asarray(x, np.float64) + 1.0) / 2.0 * clip_max


def branch_features(Xb, t_norm, cyt_idx):
    N = Xb.shape[0]
    f0 = Xb[:, 0, ..., cyt_idx]
    mask = (Xb[:, 0, ..., 6:].max(axis=-1) > 0.5)
    out = np.zeros((N, 4), np.float32)
    for i in range(N):
        f, m = f0[i], mask[i]
        out[i, 0] = (float(np.max(f)) + 1.0) / 2.0
        out[i, 1] = (float(np.mean(f)) + 1.0) / 2.0
        out[i, 2] = float(np.std(f))
        out[i, 3] = float(m.mean())
    return np.concatenate([out, t_norm.reshape(-1, 1)], axis=1)


def load_split(data_path, cyt_idx):
    Xb = np.load(data_path / "X_branch.npy").astype(np.float32)
    Xt = np.load(data_path / "X_trunk.npy").astype(np.float32)
    Y = np.load(data_path / "Y_target.npy").astype(np.float32)
    M = np.load(data_path / "Y_masks_spatial.npy").astype(np.float32)
    meta = json.loads((data_path / "metadata.json").read_text())
    clip_max = float(meta["scaling"]["max"][cyt_idx])
    dim = Y.ndim - 2
    t_norm = Xt[:, 0, -1]
    return Xb, Y, M, clip_max, dim, t_norm


def train_tf(tf, Xb, Y, dim, t_norm, cyt_idx, epochs, seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    N = Xb.shape[0]
    spatial = Y.shape[1:-1]
    P = int(np.prod(spatial))
    bfeat = branch_features(Xb, t_norm, cyt_idx)
    coords = np.array(np.meshgrid(*[np.linspace(0, 1, s) for s in spatial],
                                  indexing="ij")).reshape(len(spatial), -1).T.astype(np.float32)
    yflat = Y[..., cyt_idx].reshape(N, P, 1).astype(np.float32)
    hidden, p = 64, 32
    b_in = tf.keras.Input((5,))
    t_in = tf.keras.Input((dim,))
    b = tf.keras.layers.Dense(p)(tf.keras.layers.Dense(hidden, "relu")(b_in))
    t = tf.keras.layers.Dense(p)(tf.keras.layers.Dense(hidden, "tanh")(t_in))
    model = tf.keras.Model([b_in, t_in], [b, t])
    opt = tf.keras.optimizers.Adam(2e-3)
    bias = tf.Variable(0.0)
    n_tr = max(1, int(N * 0.7))
    coords_tf = tf.constant(coords)
    for ep in range(1, epochs + 1):
        losses = []
        for i in np.random.permutation(n_tr):
            with tf.GradientTape() as tape:
                bo, to = model([bfeat[i:i + 1], coords_tf])
                pred = tf.reduce_sum(bo * to, -1, keepdims=True) + bias
                loss = tf.reduce_mean(tf.square(pred - yflat[i]))
            vs = model.trainable_variables + [bias]
            opt.apply_gradients(zip(tape.gradient(loss, vs), vs))
            losses.append(float(loss))
        if ep % max(1, epochs // 5) == 0:
            print(f"  [tf] epoch {ep:3d}/{epochs} loss={np.mean(losses):.5f}")
    preds = np.zeros((N, P, 1), np.float32)
    for i in range(N):
        bo, to = model([bfeat[i:i + 1], coords_tf])
        preds[i] = (tf.reduce_sum(bo * to, -1, keepdims=True) + bias).numpy()
    return preds.reshape((N,) + spatial + (1,))


def train_numpy(Xb, Y, t_norm, cyt_idx, seed):
    from sklearn.linear_model import Ridge
    N = Xb.shape[0]
    spatial = Y.shape[1:-1]
    P = int(np.prod(spatial))
    bfeat = branch_features(Xb, t_norm, cyt_idx)
    Yt = Y[..., cyt_idx].reshape(N, P)
    n_tr = max(2, int(N * 0.7))
    model = Ridge(alpha=1.0).fit(bfeat[:n_tr], Yt[:n_tr])
    return model.predict(bfeat).reshape((N,) + spatial + (1,)).astype(np.float32)


def train_and_eval(data_path, cytokine, epochs, seed):
    cyt_idx = CYTOKINE_NAMES.index(cytokine)
    Xb, Y, M, clip_max, dim, t_norm = load_split(data_path, cyt_idx)
    print(f"[data] {data_path}")
    print(f"[data] dim={dim}D samples={Xb.shape[0]} spatial={Y.shape[1:-1]} "
          f"cytokine={cytokine} clip_max={clip_max:.3e}")
    try:
        import tensorflow as tf
        print("[backend] TensorFlow DeepONet")
        preds = train_tf(tf, Xb, Y, dim, t_norm, cyt_idx, epochs, seed)
        backend = "tensorflow"
    except Exception:
        print("[backend] NumPy/sklearn ridge (TF not available)")
        preds = train_numpy(Xb, Y, t_norm, cyt_idx, seed)
        backend = "numpy_ridge"
    y_true = denormalize(Y[..., cyt_idx:cyt_idx + 1], clip_max)
    y_pred = denormalize(preds, clip_max)
    metrics = calculate_metrics(y_true, y_pred, M, clip_max)
    print(f"\nDeepONet demo metrics ({cytokine}, {dim}D corner):")
    for k, v in metrics.items():
        if isinstance(v, float):
            fmt = f"{v:.4e}" if (abs(v) < 1e-3 and v != 0.0) else f"{v:.4f}"
            print(f"  {k:22s}: {fmt}")
        else:
            print(f"  {k:22s}: {v}")
    return {"cytokine": cytokine, "dim": dim, "clip_max": clip_max,
            "backend": backend, "metrics": metrics}


def run_dim(label, folder, corner, max_frames, cytokine, epochs, seed, run_dir):
    print(f"\n{label.upper()} plug & play")
    cyto_traj, mask_traj, is_2d = load_trajectory(folder, corner, max_frames)
    spatial = cyto_traj.shape[1:-1]
    print(f"[load]  {label}: {cyto_traj.shape[0]} frames -> "
          f"{'x'.join(str(s) for s in spatial)}")
    pre_dir = run_dir / f"preprocessed_{label}" / ("x".join(str(s) for s in spatial))
    preprocess(cyto_traj, mask_traj, pre_dir)
    res = train_and_eval(pre_dir, cytokine, epochs, seed)
    res["sim_folder"] = str(folder)
    (run_dir / f"results_{label}.json").write_text(json.dumps(res, indent=2))
    return res


def do_carve_demo(args):
    import pyvista as pv
    if args.sim_3d:
        carve_demo(pv, args.sim_3d, DEMO_3D, args.corner, args.demo_frames)
    if args.sim_2d:
        carve_demo(pv, args.sim_2d, DEMO_2D, args.corner, args.demo_frames)
    if not args.sim_2d and not args.sim_3d:
        sys.exit("ERROR: --carve-demo needs --sim-2d and/or --sim-3d.")
    print("\nDemo corner written. Commit demo_data/ to ship it with the repo.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dims", choices=["2d", "3d", "both"], default="both")
    ap.add_argument("--sim-2d", type=Path, default=None,
                    help="2D Step_*.vtk folder. If omitted, uses demo_data/2d.")
    ap.add_argument("--sim-3d", type=Path, default=None,
                    help="3D Step_*.vtk folder. If omitted, uses demo_data/3d.")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--corner", type=int, default=16)
    ap.add_argument("--cytokine", default="il8", choices=CYTOKINE_NAMES)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--carve-demo", action="store_true",
                    help="Carve a small corner from full data into demo_data/.")
    ap.add_argument("--demo-frames", type=int, default=12,
                    help="Frames to keep when carving the demo corner.")
    args = ap.parse_args()

    if args.carve_demo:
        do_carve_demo(args)
        return

    folder_2d = args.sim_2d or DEMO_2D
    folder_3d = args.sim_3d or DEMO_3D

    out_dir = args.out_dir or (Path(__file__).resolve().parent / "demo_out")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [f"run_{stamp}", args.cytokine, f"c{args.corner}"]
    if args.tag:
        parts.append(args.tag)
    run_dir = out_dir / "__".join(parts)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out] writing everything to: {run_dir}")

    summary = {"timestamp": stamp, "run_name": run_dir.name,
               "corner": args.corner, "cytokine": args.cytokine,
               "epochs": args.epochs, "dims": args.dims}
    if args.dims in ("3d", "both"):
        summary["3d"] = run_dim("3d", folder_3d, args.corner, args.max_frames,
                                args.cytokine, args.epochs, args.seed, run_dir)
    if args.dims in ("2d", "both"):
        summary["2d"] = run_dim("2d", folder_2d, args.corner, args.max_frames,
                                args.cytokine, args.epochs, args.seed, run_dir)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDONE. All outputs under:\n  {run_dir}")


if __name__ == "__main__":
    main()