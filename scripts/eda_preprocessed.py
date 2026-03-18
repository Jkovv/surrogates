"""
EDA on preprocessed 500x500 .npy data.
Usage:
    python scripts/eda_preprocessed.py <preprocessed/500x500 dir> [<dir2> ...]
"""
import sys, json, csv
from pathlib import Path
import numpy as np

CYTOKINES  = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
CELL_TYPES = ["EC", "NN", "NA", "M1", "M2"]


def pct(arr, q):
    return float(np.percentile(arr, q))


def eda_dir(data_dir: Path, label: str):
    print(f"\n{'='*60}")
    print(f"EDA: {label}")
    print(f"Dir: {data_dir}")
    print(f"{'='*60}")

    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)

    G        = meta["grid"]
    clip_max = meta["clip_max"]
    clip_pct = meta["scaling"]["clip_percentile"]
    print(f"Grid: {G}x{G}  |  N_timesteps: {meta['n_timesteps']}  |  N_samples: {meta['n_samples']}")

    # ── File inventory ──────────────────────────────────────────────────────
    expected = ["Y_raw_phys.npy","Y_target.npy","Y_masks_spatial.npy","Y_masks_pinn.npy",
                "X_lstm.npy","X_branch.npy","X_unet.npy","X_trunk.npy","X_colloc.npy",
                "Y_ic.npy","Y_bc_mask.npy"]
    print("\n── File inventory ──────────────────────────────────────────")
    all_ok = True
    for fname in expected:
        p = data_dir / fname
        if p.exists():
            mb = p.stat().st_size / 1e6
            arr = np.load(p, mmap_mode="r")
            print(f"  {'OK':2s}  {fname:<25s}  shape={str(arr.shape):<25s}  {mb:8.1f} MB")
        else:
            print(f"  MISSING  {fname}")
            all_ok = False
    if not all_ok:
        print("  WARNING: some files are missing!")

    # ── Raw physical cytokine stats ────────────────────────────────────────
    print("\n── Y_raw_phys — physical cytokine values ───────────────────")
    Yr = np.load(data_dir / "Y_raw_phys.npy", mmap_mode="r")  # (101,G,G,6)
    print(f"  {'Cytokine':<8} {'clip_pct':>9} {'clip_max':>14} {'global_min':>14} {'global_max':>14} {'mean':>12} {'sparsity%':>10}")
    rows_raw = []
    for j, cyt in enumerate(CYTOKINES):
        ch      = Yr[:, :, :, j].astype(np.float32)
        gmin    = float(ch.min())
        gmax    = float(ch.max())
        gmean   = float(ch.mean())
        sparsity = float(np.mean(ch == 0.0)) * 100
        print(f"  {cyt:<8} {clip_pct[j]:>9.1f} {clip_max[j]:>14.4e} {gmin:>14.4e} {gmax:>14.4e} {gmean:>12.4e} {sparsity:>9.2f}%")
        rows_raw.append({"label": label, "cytokine": cyt, "clip_pct": clip_pct[j],
                         "clip_max": clip_max[j], "global_min": gmin,
                         "global_max": gmax, "mean": gmean, "sparsity_pct": sparsity})

    # ── Scaled target stats ────────────────────────────────────────────────
    print("\n── Y_target — scaled cytokine values [-1, 1] ───────────────")
    Yt = np.load(data_dir / "Y_target.npy", mmap_mode="r")   # (99,G,G,6)
    print(f"  {'Cytokine':<8} {'min':>8} {'max':>8} {'mean':>8} {'std':>8}  {'in_range':>10}")
    for j, cyt in enumerate(CYTOKINES):
        ch = Yt[:, :, :, j].astype(np.float32)
        in_range = float(np.mean((ch >= -1.0) & (ch <= 1.0))) * 100
        print(f"  {cyt:<8} {ch.min():>8.4f} {ch.max():>8.4f} {ch.mean():>8.4f} {ch.std():>8.4f}  {in_range:>9.3f}%")

    # ── Cell-type mask stats ────────────────────────────────────────────────
    print("\n── Y_masks_spatial — cell type coverage ────────────────────")
    Ym = np.load(data_dir / "Y_masks_spatial.npy", mmap_mode="r")  # (99,G,G,5)
    print(f"  {'Cell':<6} {'mean_coverage%':>15} {'max_coverage%':>15} {'ever_present':>14}")
    for j, ct in enumerate(CELL_TYPES):
        ch = Ym[:, :, :, j].astype(np.float32)
        mean_cov = float(ch.mean()) * 100
        max_cov  = float(ch.max(axis=(1,2)).mean()) * 100
        ever     = float(ch.max()) > 0.5
        print(f"  {ct:<6} {mean_cov:>14.3f}% {max_cov:>14.3f}% {'yes' if ever else 'NO':>14}")

    # ── Temporal dynamics: per-cytokine spatial mean over time ────────────
    print("\n── Temporal dynamics (spatial mean per timestep) ───────────")
    print(f"  {'Cytokine':<8} {'t=0':>10} {'t=50':>10} {'t=98':>10} {'trend':>10}")
    for j, cyt in enumerate(CYTOKINES):
        ch = Yt[:, :, :, j].astype(np.float32)
        t0   = float(ch[0].mean())
        t50  = float(ch[50].mean())
        t98  = float(ch[-1].mean())
        trend = "↑" if t98 > t0 + 0.05 else ("↓" if t98 < t0 - 0.05 else "→")
        print(f"  {cyt:<8} {t0:>10.4f} {t50:>10.4f} {t98:>10.4f} {trend:>10}")

    # ── Boundary mask check ────────────────────────────────────────────────
    bc = np.load(data_dir / "Y_bc_mask.npy")
    n_boundary = int(bc.sum())
    expected_bc = 4 * (G - 1)
    print(f"\n── Y_bc_mask: {n_boundary} boundary cells  (expected ~{expected_bc} for {G}x{G})")

    return rows_raw


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/eda_preprocessed.py <dir1> [<dir2> ...]")
        sys.exit(1)

    all_rows = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        label = "/".join(p.parts[-4:])   # e.g. scan_iteration_0/preprocessed/500x500
        rows = eda_dir(p, label)
        all_rows.extend(rows)

    # Write summary CSV
    out = Path("eda/preprocessed_summary.csv")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    print(f"\nSummary CSV → {out}")
