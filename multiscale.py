import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from granger_test import bivariate_granger_test, select_optimal_lag


def resample_data(df, time_col, freq="W"):
    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy[time_col]):
        try:
            df_copy[time_col] = pd.to_datetime(df_copy[time_col])
        except Exception:
            return None, f"Cannot convert '{time_col}' to datetime for resampling"

    df_copy = df_copy.set_index(time_col)
    df_resampled = df_copy.resample(freq).mean()
    df_resampled = df_resampled.dropna(how="all").reset_index()
    return df_resampled, None


def multiscale_analysis(df, time_col, selected_cols, scales, max_lag=5, criterion="aic"):
    results = {}
    for scale_name, freq in scales.items():
        if freq == "original":
            resampled = df.copy()
        else:
            resampled, err = resample_data(df, time_col, freq)
            if err:
                results[scale_name] = {"error": err, "data": None, "granger": None}
                continue

        if len(resampled) < max_lag + 5:
            results[scale_name] = {
                "error": f"Insufficient data points ({len(resampled)}) after resampling",
                "data": None,
                "granger": None
            }
            continue

        try:
            granger_df = bivariate_granger_test(
                resampled, selected_cols, max_lag, criterion
            )
        except Exception as e:
            results[scale_name] = {"error": str(e), "data": None, "granger": None}
            continue

        results[scale_name] = {
            "error": None,
            "data": resampled,
            "granger": granger_df,
            "n_obs": len(resampled)
        }

    return results


def plot_multiscale_comparison(ms_results, selected_cols):
    n_scales = len(ms_results)
    if n_scales == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No multi-scale results", ha="center", va="center",
                fontsize=14, color="gray")
        return fig

    n_vars = len(selected_cols)
    fig, axes = plt.subplots(1, n_scales, figsize=(6 * n_scales, 5))
    if n_scales == 1:
        axes = [axes]

    for ax, (scale_name, result) in zip(axes, ms_results.items()):
        if result["granger"] is None:
            ax.text(0.5, 0.5, f"Error: {result.get('error', 'Unknown')}",
                    ha="center", va="center", fontsize=10, color="red",
                    transform=ax.transAxes)
            ax.set_title(scale_name, fontweight="bold")
            continue

        granger_df = result["granger"]
        matrix = np.zeros((n_vars, n_vars))
        col_idx = {col: i for i, col in enumerate(selected_cols)}

        for _, row in granger_df.iterrows():
            cause = row["cause"]
            effect = row["effect"]
            if cause in col_idx and effect in col_idx:
                val = row.get("f_statistic", 0)
                if not np.isnan(val):
                    matrix[col_idx[effect], col_idx[cause]] = val

        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        for i in range(n_vars):
            for j in range(n_vars):
                val = matrix[i, j]
                if val > 0:
                    color = "white" if val > matrix.max() * 0.6 else "black"
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                            fontsize=7, color=color)

        ax.set_xticks(range(n_vars))
        ax.set_yticks(range(n_vars))
        ax.set_xticklabels(selected_cols, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(selected_cols, fontsize=7)
        ax.set_title(f"{scale_name}\n(n={result.get('n_obs', '?')})", fontweight="bold", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Multi-scale Granger Causality Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


DEFAULT_SCALES = {
    "Original": "original",
    "Weekly": "W",
    "Monthly": "ME",
    "Quarterly": "QE"
}
