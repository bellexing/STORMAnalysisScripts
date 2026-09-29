#!/usr/bin/env python3
"""
Interactive guide-assisted centerline fitting for ONE arc-like STORM point cloud,
using the SAME centerline logic as single_arc_centerline_and_signed_hist.py.

After the user sketch is used to isolate the intended structure, the actual fit follows
the same logic as the automated spline-based script:

1) RANSAC 3-point circle fit with minimum radius constraint
2) keep points within ±band_nm radial band around the refined circle
3) define arc span from the largest continuous angular interval after cutting at the
   largest gap in sorted theta
4) fit a cubic smoothing spline r(theta) with UnivariateSpline
5) sample the centerline densely and compute signed distances

The sketch is used ONLY to choose the intended subset of points:
- points within --guide_band of the sketch are kept as candidate points
- then the automated spline-based centerline logic is run on that candidate subset
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button
from scipy.interpolate import UnivariateSpline
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

    raise ValueError("Could not infer x/y columns. Rename columns to 'Position X [nm]' and 'Position Y [nm]'.")


def load_points_csv(path: str) -> tuple[np.ndarray, str, str]:
    df = pd.read_csv(path)
    xcol, ycol = infer_xy_columns(df)
    x = pd.to_numeric(df[xcol], errors="coerce").to_numpy()
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    P = np.c_[x[m], y[m]]
    return P, xcol, ycol


def densify_polyline(points, step_nm: float = 20.0):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return pts.copy()
    out = [pts[0]]
    for i in range(len(pts) - 1):
        a = pts[i]
        b = pts[i + 1]
        seg = b - a
        L = np.linalg.norm(seg)
        if L == 0:
            continue
        n = max(2, int(np.ceil(L / step_nm)) + 1)
        for t in np.linspace(0.0, 1.0, n)[1:]:
            out.append(a + t * seg)
    return np.asarray(out, dtype=float)


def project_points_to_polyline(points_xy, polyline_xy):
    P = np.asarray(points_xy, dtype=float)
    G = np.asarray(polyline_xy, dtype=float)
    if len(G) < 2:
        raise ValueError("Need at least 2 guide points.")

    seg_starts = G[:-1]
    seg_ends = G[1:]
    seg_vecs = seg_ends - seg_starts
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    valid = seg_lens > 0
    seg_starts = seg_starts[valid]
    seg_ends = seg_ends[valid]
    seg_vecs = seg_vecs[valid]
    seg_lens = seg_lens[valid]
    if len(seg_lens) == 0:
        raise ValueError("Guide contains only zero-length segments.")

    seg_unit = seg_vecs / seg_lens[:, None]
    seg_s0 = np.concatenate([[0.0], np.cumsum(seg_lens)])[:-1]

    best_dist2 = np.full(len(P), np.inf)
    best_s = np.zeros(len(P), dtype=float)
    best_proj = np.zeros((len(P), 2), dtype=float)

    for j in range(len(seg_lens)):
        a = seg_starts[j]
        u = seg_unit[j]
        L = seg_lens[j]
        AP = P - a
        t = AP @ u
        t_clamped = np.clip(t, 0.0, L)
        proj = a + np.outer(t_clamped, u)
        d2 = np.sum((P - proj) ** 2, axis=1)
        take = d2 < best_dist2
        best_dist2[take] = d2[take]
        best_s[take] = seg_s0[j] + t_clamped[take]
        best_proj[take] = proj[take]

    return best_s, np.sqrt(best_dist2), best_proj


def circle_from_3pts(p1, p2, p3):
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    temp = x2**2 + y2**2
    bc = (x1**2 + y1**2 - temp) / 2.0
    cd = (temp - x3**2 - y3**2) / 2.0
    det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)
    if abs(det) < 1e-12:
        return None
    cx = (bc * (y2 - y3) - cd * (y1 - y2)) / det
    cy = ((x1 - x2) * cd - (x2 - x3) * bc) / det
    r = np.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2)
    return float(cx), float(cy), float(r)


def fit_circle_ransac(P: np.ndarray, thresh_nm: float = 35.0, seed: int = 0, min_radius_nm: float = 300.0):
    rng = np.random.default_rng(seed)
    n = len(P)
    if n < 20:
        raise ValueError("Too few points to fit a circle robustly.")

    idxs = np.arange(n)
    max_iter = min(15000, n * 50)
    best_inliers = None

    for _ in range(max_iter):
        i, j, k = rng.choice(idxs, size=3, replace=False)
        model = circle_from_3pts(P[i], P[j], P[k])
        if model is None:
            continue

        cx0, cy0, r0 = model
        if r0 < min_radius_nm:
            continue

        d0 = np.sqrt((P[:, 0] - cx0) ** 2 + (P[:, 1] - cy0) ** 2)
        inliers = np.abs(d0 - r0) < thresh_nm

        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    if best_inliers is None or best_inliers.sum() < 20:
        raise RuntimeError(
            "Circle RANSAC failed (not enough inliers). "
            "Try increasing --circle_thresh or lowering --min_radius."
        )

    Pin = P[best_inliers]
    xin, yin = Pin[:, 0], Pin[:, 1]
    A = np.c_[2 * xin, 2 * yin, np.ones_like(xin)]
    bvec = xin**2 + yin**2
    sol, *_ = np.linalg.lstsq(A, bvec, rcond=None)
    cx, cy, c = sol
    r = float(np.sqrt(max(cx * cx + cy * cy + c, 0.0)))

    if r < min_radius_nm:
        raise RuntimeError(
            f"Refined circle radius {r:.2f} nm is below min_radius {min_radius_nm:.2f} nm. "
            "Try lowering --min_radius, increasing --circle_thresh, or checking the ROI."
        )

    return float(cx), float(cy), float(r), best_inliers


def fit_arc_centerline(
    P: np.ndarray,
    band_nm: float = 80.0,
    circle_thresh_nm: float = 35.0,
    smooth_factor: float = 1.0,
    n_samp: int = 1000,
    seed: int = 0,
    min_radius_nm: float = 300.0,
):
    cx, cy, r, inliers = fit_circle_ransac(
        P,
        thresh_nm=circle_thresh_nm,
        seed=seed,
        min_radius_nm=min_radius_nm,
    )

    d = np.sqrt((P[:, 0] - cx) ** 2 + (P[:, 1] - cy) ** 2)
    radial_res = d - r

    keep_band = np.abs(radial_res) <= band_nm
    P_band = P[keep_band]
    d_band = d[keep_band]
    if len(P_band) < 20:
        raise RuntimeError(f"Too few points within ±{band_nm} nm band (n={len(P_band)}).")

    theta_band = np.arctan2(P_band[:, 1] - cy, P_band[:, 0] - cx)

    th = np.sort(theta_band)
    th2 = np.r_[th, th + 2 * np.pi]
    gaps = np.diff(th2)
    imax = int(np.argmax(gaps[: len(th)]))
    theta_start = float(th2[imax + 1])
    theta_end = float(theta_start + (2 * np.pi - gaps[imax]))

    theta_u = (theta_band - theta_start) % (2 * np.pi) + theta_start
    keep_arc = (theta_u >= theta_start) & (theta_u <= theta_end)

    P_arc = P_band[keep_arc]
    theta_arc = theta_u[keep_arc]
    d_arc = d_band[keep_arc]
    if len(P_arc) < 20:
        raise RuntimeError(f"Too few arc points after angle windowing (n={len(P_arc)}).")

    order = np.argsort(theta_arc)
    theta_sorted = theta_arc[order]
    r_sorted = d_arc[order]
    keep = np.r_[True, np.diff(theta_sorted) > 1e-9]
    theta_sorted = theta_sorted[keep]
    r_sorted = r_sorted[keep]

    mad = np.median(np.abs(r_sorted - np.median(r_sorted)))
    sigma_r = max(10.0, 1.4826 * mad)
    s_param = (sigma_r**2) * len(r_sorted) * smooth_factor

    spl = UnivariateSpline(theta_sorted, r_sorted, k=3, s=s_param)

    theta_s = np.linspace(theta_start, theta_end, n_samp)
    r_s = spl(theta_s)
    xc = cx + r_s * np.cos(theta_s)
    yc = cy + r_s * np.sin(theta_s)

    ds = np.hypot(np.diff(xc), np.diff(yc))
    s_cum = np.concatenate([[0.0], np.cumsum(ds)])

    tree = cKDTree(np.c_[xc, yc])
    dres, _ = tree.query(P_arc, k=1)

    return {
        "cx": cx,
        "cy": cy,
        "r": r,
        "inliers_mask": inliers,
        "band_nm": band_nm,
        "circle_thresh_nm": circle_thresh_nm,
        "theta_start": theta_start,
        "theta_end": theta_end,
        "sigma_r": sigma_r,
        "s_param": s_param,
        "P_arc": P_arc,
        "theta_arc_unwrapped": theta_arc,
        "d_arc": d_arc,
        "xc": xc,
        "yc": yc,
        "theta_s": theta_s,
        "r_s": r_s,
        "s_cum": s_cum,
        "arc_residuals": dres,
    }


def fit_arc_centerline_guided_same_logic(
    P: np.ndarray,
    guide_xy: np.ndarray,
    guide_band_nm: float = 120.0,
    band_nm: float = 80.0,
    circle_thresh_nm: float = 35.0,
    smooth_factor: float = 1.0,
    n_samp: int = 1000,
    seed: int = 0,
    min_radius_nm: float = 300.0,
):
    guide_xy = np.asarray(guide_xy, dtype=float)
    if len(guide_xy) < 2:
        raise ValueError("Need at least 2 guide points.")

    s_proj, d_perp, _ = project_points_to_polyline(P, guide_xy)
    guide_keep = d_perp <= guide_band_nm
    P_guide = P[guide_keep]

    if len(P_guide) < 20:
        raise RuntimeError(
            f"Too few points within guide band (n={len(P_guide)}). "
            "Draw the guide closer to the intended arc or increase --guide_band."
        )

    fit = fit_arc_centerline(
        P_guide,
        band_nm=band_nm,
        circle_thresh_nm=circle_thresh_nm,
        smooth_factor=smooth_factor,
        n_samp=n_samp,
        seed=seed,
        min_radius_nm=min_radius_nm,
    )
    fit["guide_band_nm"] = guide_band_nm
    fit["guide_keep_mask"] = guide_keep
    fit["n_points_near_guide"] = int(guide_keep.sum())
    return fit


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


def save_outputs(
    outdir: str,
    arc1_path: str,
    points_path: str,
    x1c: str,
    y1c: str,
    guide_pts: np.ndarray,
    fit1: dict,
    P1: np.ndarray,
    Pp: np.ndarray,
    signed: np.ndarray,
    absd: np.ndarray,
    nearest: np.ndarray,
    guide_band_nm: float,
    band_nm: float,
    min_radius: float,
    bin_w: float,
):
    os.makedirs(outdir, exist_ok=True)

    arc1_stem = safe_stem(arc1_path)
    pts_stem = safe_stem(points_path)

    guide_csv = os.path.join(outdir, f"{arc1_stem}_guide_points.csv")
    arc1_center_csv = os.path.join(outdir, f"{arc1_stem}_centerline_band{int(band_nm)}.csv")
    arc1_params_csv = os.path.join(outdir, f"{arc1_stem}_params_band{int(band_nm)}.csv")
    arc1_used_csv = os.path.join(outdir, f"{arc1_stem}_points_used_band{int(band_nm)}.csv")
    dist_csv = os.path.join(outdir, f"{pts_stem}_signed_dist_to_centerline.csv")
    overlay_png = os.path.join(outdir, f"{pts_stem}_centerline_points_signed.png")
    hist_csv = os.path.join(outdir, f"{pts_stem}_signed_dist_hist_density_bin{int(bin_w)}nm.csv")
    hist_png = os.path.join(outdir, f"{pts_stem}_signed_dist_hist_density_bin{int(bin_w)}nm.png")

    pd.DataFrame({"x_nm": guide_pts[:, 0], "y_nm": guide_pts[:, 1]}).to_csv(guide_csv, index=False)

    pd.DataFrame({
        "s_nm": fit1["s_cum"],
        "theta_rad": fit1["theta_s"],
        "x_nm": fit1["xc"],
        "y_nm": fit1["yc"],
        "r_nm": fit1["r_s"],
    }).to_csv(arc1_center_csv, index=False)

    pd.DataFrame([{
        "x_col": x1c,
        "y_col": y1c,
        "circle_center_x_nm": fit1["cx"],
        "circle_center_y_nm": fit1["cy"],
        "circle_radius_nm": fit1["r"],
        "guide_band_nm": guide_band_nm,
        "band_nm": fit1["band_nm"],
        "circle_ransac_threshold_nm": fit1["circle_thresh_nm"],
        "min_radius_nm": float(min_radius),
        "theta_start_rad": fit1["theta_start"],
        "theta_end_rad": fit1["theta_end"],
        "n_total_points": int(len(P1)),
        "n_points_near_guide": int(fit1["n_points_near_guide"]),
        "n_inliers_for_circle": int(fit1["inliers_mask"].sum()),
        "n_arc_points_used": int(len(fit1["P_arc"])),
        "sigma_r_est_nm": fit1["sigma_r"],
        "spline_s_param": fit1["s_param"],
        "arc_length_nm": float(fit1["s_cum"][-1]),
        "median_residual_nm": float(np.median(fit1["arc_residuals"])),
        "p90_residual_nm": float(np.percentile(fit1["arc_residuals"], 90)),
    }]).to_csv(arc1_params_csv, index=False)

    pd.DataFrame({
        "x_nm": fit1["P_arc"][:, 0],
        "y_nm": fit1["P_arc"][:, 1],
        "theta_rad_unwrapped": fit1["theta_arc_unwrapped"],
        "radius_nm": fit1["d_arc"],
        "radial_residual_nm": fit1["d_arc"] - fit1["r"],
    }).to_csv(arc1_used_csv, index=False)

    pd.DataFrame({
        "x_nm": Pp[:, 0],
        "y_nm": Pp[:, 1],
        "abs_dist_nm": absd,
        "signed_dist_nm": signed,
        "side": np.where(signed < 0, "concave (negative)", "convex (positive)"),
        "nearest_centerline_x_nm": nearest[:, 0],
        "nearest_centerline_y_nm": nearest[:, 1],
    }).to_csv(dist_csv, index=False)

    C = np.c_[fit1["xc"], fit1["yc"]]

    # Signed display scale from -800 to +800 nm.
    # Colour strength depends symmetrically on distance from 0:
    # -800 faint -> 0 strongest -> +800 faint.
    fixed_vmax_nm = 800.0
    signed_plot = np.clip(signed, -fixed_vmax_nm, fixed_vmax_nm)

    signed_absdist_cmap = LinearSegmentedColormap.from_list(
        "signed_absdist_green",
        [
            (0.0, "#eef8ea"),
            (0.5, "#0b5d1e"),
            (1.0, "#eef8ea"),
        ]
    )

    plt.figure(figsize=(7, 7))
    sc = plt.scatter(
        Pp[:, 0], Pp[:, 1],
        s=8,
        alpha=1,
        c=signed_plot,
        cmap=signed_absdist_cmap,
        vmin=-fixed_vmax_nm,
        vmax=fixed_vmax_nm,
    )
    plt.plot(C[:, 0], C[:, 1], linewidth=2.6, color="#9ACD32", label="Fitted mEmerald arc centerline")
    #plt.scatter([fit1["cx"]], [fit1["cy"]], s=70, marker="x", label="Circle center")
    # Keep the ORIGINAL saved-plot geometry/aspect behaviour.
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

    # Keep all four spines visible to retain the boxed appearance.
    for spine in ax.spines.values():
        spine.set_visible(True)

    plt.title("Centerline + distance-from-centreline tone map", fontsize=14)
    plt.legend(loc="upper right", fontsize=14)

    # Keep the ORIGINAL colour-bar behaviour/layout.
    cbar = plt.colorbar(sc)
    cbar.set_ticks(np.arange(-800, 801, 100))
    # Reverse the vertical colour-bar orientation:
    # 800 at the top, 0 in the middle, -800 at the bottom.
    cbar.ax.set_ylim(-fixed_vmax_nm, fixed_vmax_nm)
    cbar.set_label("Signed distance to centerline (nm)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Let Matplotlib finish the ORIGINAL layout first.
    plt.tight_layout()
    plt.gcf().canvas.draw()

    # ------------------------------------------------------------
    # L-shaped 500 nm spatial scale bar.
    #
    # Position is anchored to the FINAL axes box in axes coordinates,
    # so it stays at the visual bottom-right of the boxed plot and
    # aligns with the bottom of the colour bar.
    #
    # Its displayed length is calculated from the final x/y data spans,
    # so each arm represents exactly 500 nm.
    # ------------------------------------------------------------
    scale_bar_nm = 200.0

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

    # Compact scale label above the horizontal arm.
    ax.text(
        right_frac - x_bar_frac / 2,
        bottom_frac + 0.018,
        f"{int(scale_bar_nm)} nm",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        color="black",
        zorder=21,
        clip_on=False,
    )

    plt.savefig(overlay_png, dpi=200)
    plt.close()

    max_abs = float(np.max(np.abs(signed))) if len(signed) else 0.0
    max_abs = max(bin_w, float(np.ceil(max_abs / bin_w) * bin_w))
    edges = np.arange(-max_abs, max_abs + bin_w, bin_w)
    if len(edges) < 2:
        edges = np.array([-bin_w, 0.0, bin_w], dtype=float)

    counts, edges = np.histogram(signed, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    density = counts / (counts.sum() * bin_w) if counts.sum() > 0 else np.zeros_like(counts, dtype=float)

    pd.DataFrame({
        "bin_left_nm": edges[:-1],
        "bin_right_nm": edges[1:],
        "bin_center_nm": centers,
        "count": counts,
        "density_per_nm": density,
    }).to_csv(hist_csv, index=False)

    plt.figure(figsize=(7, 4))
    plt.hist(signed, bins=edges, density=True if len(signed) else False)
    plt.axvline(0, linewidth=1)
    plt.xlabel("Signed distance to centerline (nm)")
    plt.ylabel("Density (1/nm)")
    plt.title(f"Signed distance distribution ({bin_w:g} nm bins)")
    plt.tight_layout()
    plt.savefig(hist_png, dpi=200)
    plt.close()

    return {
        "guide_csv": guide_csv,
        "centerline_csv": arc1_center_csv,
        "params_csv": arc1_params_csv,
        "used_csv": arc1_used_csv,
        "dist_csv": dist_csv,
        "overlay_png": overlay_png,
        "hist_csv": hist_csv,
        "hist_png": hist_png,
    }


class GuidedFitApp:
    def __init__(self, arc1_path, points_path, outdir, guide_band, band, circle_thresh, min_radius, smooth_factor, n_samp, bin_w):
        self.arc1_path = arc1_path
        self.points_path = points_path
        self.outdir = outdir

        self.guide_band = float(guide_band)
        self.band = float(band)
        self.circle_thresh = float(circle_thresh)
        self.min_radius = float(min_radius)
        self.smooth_factor = float(smooth_factor)
        self.n_samp = int(n_samp)
        self.bin_w = float(bin_w)

        self.P1, self.x1c, self.y1c = load_points_csv(arc1_path)
        self.Pp, _, _ = load_points_csv(points_path)

        self.guide_pts = []
        self.fit = None
        self.last_signed = None
        self.last_absd = None
        self.last_nearest = None
        self.saved_paths = None

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        plt.subplots_adjust(bottom=0.16)

        self.ax.scatter(self.P1[:, 0], self.P1[:, 1], s=2, alpha=0.45, label="Arc localizations")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlabel("x (nm)")
        self.ax.set_ylabel("y (nm)")
        self.ax.set_title("Draw rough guide, then press Enter to fit.")
        self.ax.legend(loc="best")

        self.guide_line, = self.ax.plot([], [], "o-", linewidth=2, markersize=5, label="Guide")
        self.center_line, = self.ax.plot([], [], "-", linewidth=2.5, label="Fitted centerline")
        self.center_mark = self.ax.scatter([], [], s=70, marker="x", label="Circle center")
        self.used_scatter = self.ax.scatter([], [], s=7, alpha=0.25, label="Arc points used")

        ax_fit = plt.axes([0.10, 0.04, 0.14, 0.07])
        ax_reset = plt.axes([0.28, 0.04, 0.14, 0.07])
        ax_save = plt.axes([0.46, 0.04, 0.14, 0.07])
        ax_quit = plt.axes([0.64, 0.04, 0.14, 0.07])

        self.btn_fit = Button(ax_fit, "Fit")
        self.btn_reset = Button(ax_reset, "Reset")
        self.btn_save = Button(ax_save, "Save")
        self.btn_quit = Button(ax_quit, "Quit")

        self.btn_fit.on_clicked(lambda event: self.fit_now())
        self.btn_reset.on_clicked(lambda event: self.reset())
        self.btn_save.on_clicked(lambda event: self.save_now())
        self.btn_quit.on_clicked(lambda event: plt.close(self.fig))

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            self.guide_pts.append([event.xdata, event.ydata])
            self.update_guide()
        elif event.button == 3 and self.guide_pts:
            self.guide_pts.pop()
            self.update_guide()

    def on_key(self, event):
        if event.key == "enter":
            self.fit_now()
        elif event.key == "r":
            self.reset()
        elif event.key == "s":
            self.save_now()
        elif event.key == "q":
            plt.close(self.fig)

    def update_guide(self):
        if len(self.guide_pts) == 0:
            self.guide_line.set_data([], [])
        else:
            g = np.asarray(self.guide_pts, dtype=float)
            self.guide_line.set_data(g[:, 0], g[:, 1])
        self.fig.canvas.draw_idle()

    def fit_now(self):
        if len(self.guide_pts) < 2:
            self.ax.set_title("Need at least 2 guide points.")
            self.fig.canvas.draw_idle()
            return
        try:
            guide_dense = densify_polyline(self.guide_pts, step_nm=max(10.0, self.band / 2.0))
            self.fit = fit_arc_centerline_guided_same_logic(
                self.P1,
                guide_dense,
                guide_band_nm=self.guide_band,
                band_nm=self.band,
                circle_thresh_nm=self.circle_thresh,
                smooth_factor=self.smooth_factor,
                n_samp=self.n_samp,
                seed=1,
                min_radius_nm=self.min_radius,
            )

            C = np.c_[self.fit["xc"], self.fit["yc"]]
            self.last_signed, self.last_absd, self.last_nearest = signed_distance_to_centerline(
                self.Pp, C, np.array([self.fit["cx"], self.fit["cy"]], dtype=float)
            )

            self.center_line.set_data(self.fit["xc"], self.fit["yc"])
            self.center_mark.set_offsets(np.array([[self.fit["cx"], self.fit["cy"]]], dtype=float))
            self.used_scatter.set_offsets(self.fit["P_arc"])

            self.ax.set_title(
                "Fit complete: guide-selected subset + same spline logic as automated script. "
                f"Used {len(self.fit['P_arc'])} arc points."
            )
            self.fig.canvas.draw_idle()
        except Exception as e:
            self.ax.set_title(f"Fit failed: {e}")
            self.fig.canvas.draw_idle()

    def reset(self):
        self.guide_pts = []
        self.fit = None
        self.last_signed = None
        self.last_absd = None
        self.last_nearest = None
        self.saved_paths = None
        self.guide_line.set_data([], [])
        self.center_line.set_data([], [])
        self.center_mark.set_offsets(np.empty((0, 2)))
        self.used_scatter.set_offsets(np.empty((0, 2)))
        self.ax.set_title("Draw rough guide, then press Enter to fit.")
        self.fig.canvas.draw_idle()

    def save_now(self):
        if self.fit is None or self.last_signed is None:
            self.ax.set_title("Nothing fitted yet. Draw guide and press Enter first.")
            self.fig.canvas.draw_idle()
            return

        self.saved_paths = save_outputs(
            outdir=self.outdir,
            arc1_path=self.arc1_path,
            points_path=self.points_path,
            x1c=self.x1c,
            y1c=self.y1c,
            guide_pts=np.asarray(self.guide_pts, dtype=float),
            fit1=self.fit,
            P1=self.P1,
            Pp=self.Pp,
            signed=self.last_signed,
            absd=self.last_absd,
            nearest=self.last_nearest,
            guide_band_nm=self.guide_band,
            band_nm=self.band,
            min_radius=self.min_radius,
            bin_w=self.bin_w,
        )

        self.ax.set_title("Saved outputs to: " + self.outdir)
        self.fig.canvas.draw_idle()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arc1", required=True, help="CSV with Arc 1 localizations (x/y in nm).")
    ap.add_argument("--points", required=True, help="CSV with points to score (often same as arc1).")
    ap.add_argument("--outdir", default=".", help="Output directory.")
    ap.add_argument("--guide_band", type=float, default=120.0, help="Distance (nm) from drawn guide for candidate arc points.")
    ap.add_argument("--band", type=float, default=80.0, help="Radial band (nm) around the reference circle.")
    ap.add_argument("--circle_thresh", type=float, default=35.0, help="RANSAC inlier threshold (nm) for circle fit.")
    ap.add_argument("--min_radius", type=float, default=300.0, help="Minimum allowed circle radius (nm).")
    ap.add_argument("--smooth_factor", type=float, default=1.0, help="Spline smoothing factor (larger => smoother).")
    ap.add_argument("--n_samp", type=int, default=1000, help="Number of samples along centerline.")
    ap.add_argument("--bin_w", type=float, default=5.0, help="Histogram bin width (nm).")
    args = ap.parse_args()

    app = GuidedFitApp(
        arc1_path=args.arc1,
        points_path=args.points,
        outdir=args.outdir,
        guide_band=args.guide_band,
        band=args.band,
        circle_thresh=args.circle_thresh,
        min_radius=args.min_radius,
        smooth_factor=args.smooth_factor,
        n_samp=args.n_samp,
        bin_w=args.bin_w,
    )
    plt.show()

    if app.saved_paths is not None:
        print("Saved:")
        print("  Guide points:", app.saved_paths["guide_csv"])
        print("  Centerline:", app.saved_paths["centerline_csv"])
        print("  Params:", app.saved_paths["params_csv"])
        print("  Points used:", app.saved_paths["used_csv"])
        print("  Per-point distances:", app.saved_paths["dist_csv"])
        print("  Overlay plot:", app.saved_paths["overlay_png"])
        print("  Histogram plot:", app.saved_paths["hist_png"])
        print("  Histogram CSV:", app.saved_paths["hist_csv"])
    else:
        print("No outputs saved.")


if __name__ == "__main__":
    main()
