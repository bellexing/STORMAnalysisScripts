#!/usr/bin/env python3
"""
Constant-baseline subtraction + single-Gaussian fitting for two histogram CSVs
and an overlay plot.

Input CSVs must contain:
  - bin_center_nm
  - density_per_nm

What it does:
  1) Estimate a constant baseline from the outer |x| tails (top 20% |x|)
  2) Subtract baseline, clamp negatives to 0
  3) Fit a single Gaussian: A * exp(-(x-mu)^2/(2*sigma^2))
  4) Plot both fitted Gaussians on one graph
     - 568_table: green
     - 647_table: magenta
     - vertical dotted lines mark mu (peak center)
  5) Calculate the peak shift:
     shift_nm = mu_647 - mu_568
     abs_shift_nm = |mu_647 - mu_568|
  6) Save:
     - single_gauss_fits_params.csv/.xlsx
     - single_gauss_fits_curves.csv/.xlsx
     - single_gauss_peak_shift_summary.csv/.xlsx
     - signed_distance_hist_single_gauss_overlay.png

Dependencies:
  pip install numpy pandas scipy matplotlib openpyxl
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def estimate_baseline_constant(x, y):
    """
    Constant baseline estimate:
    - take the bins in the far tails: top 20% of |x|
    - baseline = median density in those tails
    """
    ax = np.abs(x)
    thr = np.quantile(ax, 0.80)
    tail = y[ax >= thr]
    if len(tail) < 5:
        # fallback: median of the lowest 10% y values
        tail = np.sort(y)[: max(5, int(0.10 * len(y)))]
    return float(np.median(tail))


def fit_single_gaussian(x, y_corr):
    """
    Fit single Gaussian to baseline-corrected data.
    """
    A0 = float(np.max(y_corr))
    mu0 = float(x[np.argmax(y_corr)])

    w = y_corr / (np.sum(y_corr) + 1e-12)
    sigma0 = float(np.sqrt(np.sum(w * (x - mu0) ** 2)))
    sigma0 = max(sigma0, 5.0)

    bounds = ([0.0, float(np.min(x)), 1e-3], [np.inf, float(np.max(x)), np.inf])
    popt, _ = curve_fit(
        gaussian,
        x,
        y_corr,
        p0=[A0, mu0, sigma0],
        bounds=bounds,
        maxfev=20000,
    )
    return popt  # A, mu, sigma


def r2_score(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv568", required=True, help="Histogram CSV for 568_table")
    ap.add_argument("--csv647", required=True, help="Histogram CSV for 647_table")
    ap.add_argument("--outdir", default=".", help="Output directory")
    ap.add_argument("--title", default="Single-Gaussian fits", help="Base plot title")
    ap.add_argument("--bin_col", default="bin_center_nm", help="Column name for x")
    ap.add_argument("--dens_col", default="density_per_nm", help="Column name for y")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    inputs = {
        "568_table": args.csv568,
        "647_table": args.csv647,
    }
    colors = {
        "568_table": "green",
        "647_table": "magenta",
    }

    series = {}
    fit_rows = []

    # ----- Load, baseline-correct, fit -----
    for name, path in inputs.items():
        df = pd.read_csv(path)

        if args.bin_col not in df.columns or args.dens_col not in df.columns:
            raise ValueError(
                f"{name}: expected columns '{args.bin_col}' and '{args.dens_col}' in {path}"
            )

        x = pd.to_numeric(df[args.bin_col], errors="coerce").to_numpy()
        y = pd.to_numeric(df[args.dens_col], errors="coerce").to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]

        baseline = estimate_baseline_constant(x, y)
        y_corr = y - baseline
        y_corr[y_corr < 0] = 0.0

        if np.max(y_corr) <= 0:
            raise ValueError(f"{name}: no positive signal remains after baseline subtraction.")

        popt = fit_single_gaussian(x, y_corr)
        A, mu, sigma = map(float, popt)

        yhat = gaussian(x, *popt)
        r2 = r2_score(y_corr, yhat)

        fit_rows.append(
            {
                "dataset": name,
                "input_file": os.path.basename(path),
                "baseline_density_per_nm": baseline,
                "A_peak_density_per_nm": A,
                "mu_peak_center_nm": mu,
                "sigma_nm": sigma,
                "FWHM_nm": 2.354820045 * sigma,
                "R2_on_corrected": r2,
            }
        )

        series[name] = {
            "x": x,
            "y_raw": y,
            "y_corr": y_corr,
            "popt": popt,
        }

    # ----- Calculate peak shift -----
    mu_568 = float(series["568_table"]["popt"][1])
    mu_647 = float(series["647_table"]["popt"][1])
    shift_nm = mu_647 - mu_568
    abs_shift_nm = abs(shift_nm)

    # ----- Save parameter table -----
    fit_df = pd.DataFrame(fit_rows).sort_values("dataset")
    fit_df["mu_568_nm"] = mu_568
    fit_df["mu_647_nm"] = mu_647
    fit_df["peak_shift_647_minus_568_nm"] = shift_nm
    fit_df["peak_shift_absolute_nm"] = abs_shift_nm

    out_params_csv = os.path.join(args.outdir, "single_gauss_fits_params.csv")
    out_params_xlsx = os.path.join(args.outdir, "single_gauss_fits_params.xlsx")
    fit_df.to_csv(out_params_csv, index=False)
    try:
        fit_df.to_excel(out_params_xlsx, index=False)
    except Exception:
        pass

    # ----- Save shift summary -----
    shift_df = pd.DataFrame(
        [
            {
                "mu_568_nm": mu_568,
                "mu_647_nm": mu_647,
                "peak_shift_647_minus_568_nm": shift_nm,
                "peak_shift_absolute_nm": abs_shift_nm,
            }
        ]
    )

    out_shift_csv = os.path.join(args.outdir, "single_gauss_peak_shift_summary.csv")
    out_shift_xlsx = os.path.join(args.outdir, "single_gauss_peak_shift_summary.xlsx")
    shift_df.to_csv(out_shift_csv, index=False)
    try:
        shift_df.to_excel(out_shift_xlsx, index=False)
    except Exception:
        pass

    # ----- Save per-bin curves table -----
    merged = None
    for name in ["568_table", "647_table"]:
        x = series[name]["x"]
        popt = series[name]["popt"]
        dfi = pd.DataFrame(
            {
                "bin_center_nm": x,
                f"{name}_density_per_nm": series[name]["y_raw"],
                f"{name}_density_corr_per_nm": series[name]["y_corr"],
                f"{name}_fit_per_nm": gaussian(x, *popt),
            }
        )
        merged = dfi if merged is None else pd.merge(merged, dfi, on="bin_center_nm", how="outer")

    merged = merged.sort_values("bin_center_nm")
    out_curves_csv = os.path.join(args.outdir, "single_gauss_fits_curves.csv")
    out_curves_xlsx = os.path.join(args.outdir, "single_gauss_fits_curves.xlsx")
    merged.to_csv(out_curves_csv, index=False)
    try:
        merged.to_excel(out_curves_xlsx, index=False)
    except Exception:
        pass

    # ----- Plot overlay -----
    out_png = os.path.join(args.outdir, "signed_distance_hist_single_gauss_overlay.png")
    plt.figure(figsize=(7.8, 5.0))

    x_all = np.concatenate([series["568_table"]["x"], series["647_table"]["x"]])
    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    xs = np.linspace(x_min, x_max, 2000)

    ymax = 0.0

    # Draw baseline-corrected histogram data as coloured bars.
    # Infer bin widths independently for each channel from the x coordinates.
    for name in ["568_table", "647_table"]:
        x = series[name]["x"]
        y_corr = series[name]["y_corr"]
        popt = series[name]["popt"]
        mu = float(popt[1])
        c = colors[name]

        if len(x) > 1:
            dx = np.diff(np.sort(np.unique(x)))
            bin_width = float(np.median(dx)) if len(dx) > 0 else 5.0
        else:
            bin_width = 5.0

        yfit = gaussian(xs, *popt)
        ymax = max(ymax, float(np.max(yfit)), float(np.max(y_corr)))

        # Histogram bars: green for 568, magenta for 647
        plt.bar(
            x,
            y_corr,
            width=bin_width * 0.9,
            color=c,
            alpha=0.28,
            edgecolor="none",
            label=("mEmerald histogram" if name == "568_table" else "mScarlet histogram"),
        )

        # Gaussian fit curve
        plt.plot(
            xs,
            yfit,
            linewidth=2.5,
            color=c,
            label=("mEmerald Gaussian fit" if name == "568_table" else "mScarlet Gaussian fit"),
        )

        # Peak-centre marker, annotation only (no fitted numeric value)
        plt.axvline(
            mu,
            linestyle=":",
            linewidth=2.0,
            color=c,
        )

        # Peak-centre text annotations are hidden for now, but the code is kept
        # here in case you want to re-enable them later.
        # peak_y = gaussian(mu, *popt)
        # label_text = "mEmerald peak centre" if name == "568_table" else "mScarlet peak centre"
        # plt.annotate(
        #     label_text,
        #     xy=(mu, peak_y),
        #     xytext=(6, 10),
        #     textcoords="offset points",
        #     color=c,
        #     fontsize=9,
        #     ha="left",
        #     va="bottom",
        # )

    plt.xlabel("Signed distance (nm)", fontsize=12)
    plt.ylabel("Normalised localisation density", fontsize=12)
    plt.title(args.title, fontsize=12)
    # Peak-shift annotation without displaying the numerical value.
    y_line = ymax * 0.92 if ymax > 0 else 1.0
    plt.annotate(
        "",
        xy=(mu_647, y_line),
        xytext=(mu_568, y_line),
        arrowprops=dict(arrowstyle="<->", linewidth=1.5),
    )
    plt.text(
        (mu_568 + mu_647) / 2.0,
        y_line,
        "Peak shift",
        va="bottom",
        ha="center",
        fontsize=10,
    )

    handles, labels = plt.gca().get_legend_handles_labels()
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            h2.append(h)
            l2.append(l)
    plt.legend(h2, l2, loc="upper right", fontsize=10)

    plt.tick_params(axis="both", labelsize=10)
    plt.xlim(-800, 800)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print("Saved:")
    print("  Plot:", out_png)
    print("  Params:", out_params_csv)
    print("  Curves:", out_curves_csv)
    print("  Shift summary:", out_shift_csv)

    print("\nPeak centers (mu):")
    print(f"  568_table: mu = {mu_568:.3f} nm")
    print(f"  647_table: mu = {mu_647:.3f} nm")
    print(f"\nPeak shift:")
    print(f"  647 - 568 = {shift_nm:.3f} nm")
    print(f"  |shift|    = {abs_shift_nm:.3f} nm")


if __name__ == "__main__":
    main()