# analyze_llie_pairs.py
import os, glob, argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

def rgb_to_luma_np(im):  # im: HxWx3 float [0,1]
    return 0.299*im[...,0] + 0.587*im[...,1] + 0.114*im[...,2]

def laplacian_var(luma):
    # simple Laplacian kernel
    k = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
    p = np.pad(luma, ((1,1),(1,1)), mode='reflect')
    out = (
        k[0,0]*p[:-2,:-2] + k[0,1]*p[:-2,1:-1] + k[0,2]*p[:-2,2:] +
        k[1,0]*p[1:-1,:-2] + k[1,1]*p[1:-1,1:-1] + k[1,2]*p[1:-1,2:] +
        k[2,0]*p[2:,:-2] + k[2,1]*p[2:,1:-1] + k[2,2]*p[2:,2:]
    )
    return float(out.var())

def read_rgb(path):
    im = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return im

def per_image_stats(im):
    l = rgb_to_luma_np(im)
    q = np.quantile(l, [0.01,0.05,0.10,0.50,0.90,0.95,0.99])
    rgb_mean = im.mean(axis=(0,1))
    rgb_std  = im.std(axis=(0,1))
    eps = 1e-6
    return {
        "H": im.shape[0], "W": im.shape[1],
        "l_mean": float(l.mean()),
        "l_std": float(l.std()),
        "l_p01": float(q[0]), "l_p05": float(q[1]), "l_p10": float(q[2]),
        "l_p50": float(q[3]), "l_p90": float(q[4]), "l_p95": float(q[5]), "l_p99": float(q[6]),
        "dark_frac_005": float((l < 0.05).mean()),
        "dark_frac_010": float((l < 0.10).mean()),
        "clip_frac_098": float((l > 0.98).mean()),
        "r_mean": float(rgb_mean[0]), "g_mean": float(rgb_mean[1]), "b_mean": float(rgb_mean[2]),
        "r_std": float(rgb_std[0]),  "g_std": float(rgb_std[1]),  "b_std": float(rgb_std[2]),
        "rg": float((rgb_mean[0]+eps)/(rgb_mean[1]+eps)),
        "bg": float((rgb_mean[2]+eps)/(rgb_mean[1]+eps)),
        "sharp_lap_var": laplacian_var(l),
    }

def save_csv(rows, out_csv):
    import csv
    keys = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--low_dir", required=True)
    ap.add_argument("--normal_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    low_paths = sorted(glob.glob(os.path.join(args.low_dir, "*.jpg")))
    nor_paths = sorted(glob.glob(os.path.join(args.normal_dir, "*.jpg")))
    low_names = [os.path.basename(p) for p in low_paths]
    nor_names = [os.path.basename(p) for p in nor_paths]

    # Pair check
    common = sorted(set(low_names) & set(nor_names))
    print(f"low: {len(low_paths)}, normal: {len(nor_paths)}, paired(common): {len(common)}")
    missing_low = sorted(set(nor_names) - set(low_names))
    missing_nor = sorted(set(low_names) - set(nor_names))
    if missing_low: print("Missing in low:", missing_low[:10], "...")
    if missing_nor: print("Missing in normal:", missing_nor[:10], "...")

    rows_low, rows_nor, rows_pair = [], [], []
    for name in tqdm(common):
        lp = os.path.join(args.low_dir, name)
        npth = os.path.join(args.normal_dir, name)

        low = read_rgb(lp)
        nor = read_rgb(npth)

        sL = per_image_stats(low); sL["file"] = name
        sN = per_image_stats(nor); sN["file"] = name

        # simple pair-level indicators
        l_low = sL["l_p50"]; l_nor = sN["l_p50"]
        exposure_ratio = float((l_nor + 1e-6) / (l_low + 1e-6))  # how much brighter GT is vs input (median luma)
        mae = float(np.mean(np.abs(low - nor)))

        sP = {
            "file": name,
            "H": sL["H"], "W": sL["W"],
            "low_l_p50": sL["l_p50"], "nor_l_p50": sN["l_p50"],
            "exposure_ratio_med": exposure_ratio,
            "mae_rgb": mae,
        }

        rows_low.append(sL)
        rows_nor.append(sN)
        rows_pair.append(sP)

    save_csv(rows_low, os.path.join(args.out_dir, "stats_low.csv"))
    save_csv(rows_nor, os.path.join(args.out_dir, "stats_normal.csv"))
    save_csv(rows_pair, os.path.join(args.out_dir, "stats_pair.csv"))

    # Plots
    def hist_plot(vals, title, fname):
        plt.figure()
        plt.hist(vals, bins=50)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, fname), dpi=200)
        plt.close()

    low_lmean = [r["l_mean"] for r in rows_low]
    nor_lmean = [r["l_mean"] for r in rows_nor]
    ratio = [r["exposure_ratio_med"] for r in rows_pair]
    dark005 = [r["dark_frac_005"] for r in rows_low]

    hist_plot(low_lmean, "Low luma mean", "hist_low_lmean.png")
    hist_plot(nor_lmean, "Normal luma mean", "hist_normal_lmean.png")
    hist_plot(ratio, "Exposure ratio (median luma normal/low)", "hist_exposure_ratio.png")
    hist_plot(dark005, "Low dark fraction (luma<0.05)", "hist_low_dark005.png")

    # Scatter: low vs normal luma mean
    plt.figure()
    plt.scatter(low_lmean, nor_lmean, s=8)
    plt.xlabel("low luma mean")
    plt.ylabel("normal luma mean")
    plt.title("Low vs Normal luma mean")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "scatter_low_vs_normal_lmean.png"), dpi=200)
    plt.close()

    print("Saved to:", args.out_dir)

if __name__ == "__main__":
    main()
