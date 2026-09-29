#!/usr/bin/env python3
import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree


def safe_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "", stem)
    return stem


def infer_xy_columns(df: pd.DataFrame) -> tuple[str, str]:
    candidates = [
        ("Position X [nm]", "Position Y [nm]"),
        ("x [nm]", "y [nm]"),
        ("x_nm", "y_nm"),
        ("x", "y"),
    ]
    for a, b in candidates:
        if a in df.columns and b in df.columns:
            return a, b

    good = []
    for c in df.columns:
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy()
        if np.isfinite(vals).sum() > 20:
            good.append(c)

    if len(good) >= 2:
        return good[0], good[1]

    raise ValueError("Could not infer x/y columns.")


def load_points_csv(path: str) -> tuple[np.ndarray, str, str]:
    df = pd.read_csv(path)
    xcol, ycol = infer_xy_columns(df)
    x = pd.to_numeric(df[xcol], errors="coerce").to_numpy()
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    P = np.c_[x[m], y[m]]
    return P, xcol, ycol


def load_centerline_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    if "x_nm" in df.columns and "y_nm" in df.columns:
        return df[["x_nm", "y_nm"]].to_numpy(dtype=float)
    elif "x" in df.columns and "y" in df.columns:
        return df[["x", "y"]].to_numpy(dtype=float)
    else:
        raise ValueError("Centerline CSV must contain x_nm/y_nm or x/y columns.")


def load_circle_center(params_csv: str) -> np.ndarray:
    df = pd.read_csv(params_csv)
    row = df.iloc[0]
    return np.array([
        float(row["circle_center_x_nm"]),
        float(row["circle_center_y_nm"]),
    ], dtype=float)


def signed_distance_to_centerline(P: np.ndarray, C: np.ndarray, circle_center: np.ndarray):
    tree = cKDTree(C)
    dist, idx = tree.query(P, k=1)
    nearest = C[idx]

    v = P - nearest
    to_center = circle_center[None, :] - nearest
    dot = np.einsum("ij,ij->i", v, to_center)

    signed = dist.copy()
    signed[dot > 0] *= -1
    return signed, dist, nearest


def robust_absdist_vmax(absd: np.ndarray, clip_percentile: float = 95.0) -> float:
    absd = np.asarray(absd, dtype=float)
    absd = absd[np.isfinite(absd)]
    if absd.size == 0:
        return 1.0
    vmax = float(np.percentile(absd, clip_percentile))
    return max(vmax, 1.0)


def make_magenta_cmap():
    return LinearSegmentedColormap.from_list(
        "custom_magenta",
        ["#fde6f2", "#f6b6da", "#ec73c0", "#d62d9a", "#8e0152"]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--centerline", required=True, help="Saved centerline CSV")
    ap.add_argument("--params", required=True, help="Saved params CSV with circle center")
    ap.add_argument("--points", required=True, help="New points CSV to score")
    ap.add_argument("--outdir", default=".", help="Output directory")
    ap.add_argument("--bin_w", type=float, default=5.0, help="Histogram bin width (nm)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    C = load_centerline_csv(args.centerline)
    circle_center = load_circle_center(args.params)
    P, xcol, ycol = load_points_csv(args.points)

    signed, absd, nearest = signed_distance_to_centerline(P, C, circle_center)

    pts_stem = safe_stem(args.points)

    dist_csv = os.path.join(args.outdir, f"{pts_stem}_signed_dist_to_saved_centerline.csv")
    overlay_png = os.path.join(args.outdir, f"{pts_stem}_saved_centerline_points_signed.png")
    hist_csv = os.path.join(args.outdir, f"{pts_stem}_signed_dist_hist_density_bin{int(args.bin_w)}nm.csv")
    hist_png = os.path.join(args.outdir, f"{pts_stem}_signed_dist_hist_density_bin{int(args.bin_w)}nm.png")

    pd.DataFrame({
        "x_nm": P[:, 0],
        "y_nm": P[:, 1],
        "abs_dist_nm": absd,
        "signed_dist_nm": signed,
        "side": np.where(signed < 0, "concave (negative)", "convex (positive)"),
        "nearest_centerline_x_nm": nearest[:, 0],
        "nearest_centerline_y_nm": nearest[:, 1],
    }).to_csv(dist_csv, index=False)
    
    
    # Signed display scale from 800 to -800 nm.
    # Colour strength is symmetric around 0:
    # 800 faint -> 0 strongest magenta -> -800 faint.
    fixed_vmax_nm = 800.0
    signed_plot = np.clip(signed, -fixed_vmax_nm, fixed_vmax_nm)

    signed_magenta_cmap = LinearSegmentedColormap.from_list(
        "signed_magenta",
        [
            (0.0, "#fde6f2"),   # -800 nm -> faint
            (0.5, "#8e0152"),   #    0 nm -> strongest
            (1.0, "#fde6f2"),   # +800 nm -> faint
        ]
    )

    plt.figure(figsize=(7, 7))
    sc = plt.scatter(
        P[:, 0], P[:, 1],
        s=8,
        alpha=1,
        c=signed_plot,
        cmap=signed_magenta_cmap,
        vmin=-fixed_vmax_nm,
        vmax=fixed_vmax_nm,
    )
    plt.plot(
        C[:, 0], C[:, 1],
        linewidth=5,
        color="#9ACD32",
        label="Fitted mEmerald arc centerline",
    )
    # No circle-centre marker is drawn.

    # Keep the ORIGINAL plot geometry/aspect behaviour.
    plt.axis("equal")

    # Hide x/y coordinate labels and tick values, but keep the original box.
    ax = plt.gca()
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )

    plt.legend(loc="upper right", fontsize=14)

    # Keep the ORIGINAL colour-bar behaviour/layout.
    cbar = plt.colorbar(sc)
    cbar.set_ticks(np.arange(-800, 801, 100))
    cbar.ax.set_ylim(-fixed_vmax_nm, fixed_vmax_nm)  # top 800, bottom -800
    cbar.set_label("Signed distance to centerline (nm)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Let Matplotlib finish the ORIGINAL layout first.
    plt.tight_layout()
    plt.gcf().canvas.draw()

    # ------------------------------------------------------------
    # L-shaped 500 nm spatial scale bar.
    #
    # IMPORTANT:
    # Position is anchored to the FINAL axes box in axes coordinates,
    # so it stays right at the visual bottom-right of the boxed plot,
    # aligned with the bottom of the colour bar.
    #
    # Its displayed length is still calculated from the final x/y data
    # spans, so each arm represents exactly 500 nm.
    # ------------------------------------------------------------
    scale_bar_nm = 100.0

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xspan = xmax - xmin
    yspan = ymax - ymin

    # Convert 500 nm into fractions of the final axes width/height.
    x_bar_frac = scale_bar_nm / xspan
    y_bar_frac = scale_bar_nm / yspan

    # Anchor very close to the bottom-right border of the plotting box.
    right_frac = 0.965
    bottom_frac = 0.025

    # Horizontal 500 nm arm.
    ax.plot(
        [right_frac - x_bar_frac, right_frac],
        [bottom_frac, bottom_frac],
        transform=ax.transAxes,
        color="black",
        linewidth=3,
        solid_capstyle="butt",
        zorder=20,
        clip_on=False,
    )

    # Vertical 500 nm arm.
    ax.plot(
        [right_frac, right_frac],
        [bottom_frac, bottom_frac + y_bar_frac],
        transform=ax.transAxes,
        color="black",
        linewidth=3,
        solid_capstyle="butt",
        zorder=20,
        clip_on=False,
    )

    # Compact label above the horizontal arm.
    #ax.text(
    #    right_frac - x_bar_frac / 2,
    #    bottom_frac + 0.018,
    #    f"{int(scale_bar_nm)} nm",
    #    transform=ax.transAxes,
    #    ha="center",
    #    va="bottom",
    #    fontsize=12,
    #    color="black",
    #    zorder=21,
    #    clip_on=False,
    #)

    plt.savefig(overlay_png, dpi=200)
    plt.close()

    bin_w = float(args.bin_w)
    max_abs = float(np.ceil(np.max(np.abs(signed)) / bin_w) * bin_w)
    edges = np.arange(-max_abs, max_abs + bin_w, bin_w)

    counts, edges = np.histogram(signed, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    density = counts / (counts.sum() * bin_w)

    pd.DataFrame({
        "bin_left_nm": edges[:-1],
        "bin_right_nm": edges[1:],
        "bin_center_nm": centers,
        "count": counts,
        "density_per_nm": density,
    }).to_csv(hist_csv, index=False)

    plt.figure(figsize=(7, 4))
    plt.hist(signed, bins=edges, density=True)
    plt.axvline(0, linewidth=1)
    plt.xlabel("Signed distance to centerline (nm)")
    plt.ylabel("Density (1/nm)")
    plt.title(f"Signed distance distribution ({bin_w:g} nm bins)")
    plt.tight_layout()
    plt.savefig(hist_png, dpi=200)
    plt.close()

    print("Saved:")
    print("  Per-point distances:", dist_csv)
    print("  Overlay plot:", overlay_png)
    print("  Histogram plot:", hist_png)
    print("  Histogram CSV:", hist_csv)


if __name__ == "__main__":
    main()