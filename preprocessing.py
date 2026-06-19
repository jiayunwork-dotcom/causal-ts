import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def adf_test(series, significance=0.05):
    series_clean = series.dropna()
    if len(series_clean) < 10:
        return {
            "adf_statistic": np.nan,
            "p_value": np.nan,
            "used_lag": np.nan,
            "n_obs": len(series_clean),
            "is_stationary": None,
            "warning": "Too few observations for ADF test"
        }
    result = adfuller(series_clean, autolag="AIC")
    return {
        "adf_statistic": round(result[0], 4),
        "p_value": round(result[1], 4),
        "used_lag": result[2],
        "n_obs": result[3],
        "is_stationary": result[1] < significance,
        "critical_values": {k: round(v, 4) for k, v in result[4].items()},
        "warning": None
    }


def run_adf_tests(df, selected_cols, significance=0.05):
    results = []
    for col in selected_cols:
        res = adf_test(df[col], significance)
        results.append({
            "variable": col,
            "adf_statistic": res["adf_statistic"],
            "p_value": res["p_value"],
            "used_lag": res.get("used_lag", np.nan),
            "n_obs": res.get("n_obs", np.nan),
            "is_stationary": res["is_stationary"],
            "warning": res["warning"]
        })
    return pd.DataFrame(results), results


def apply_differencing(df, selected_cols, order=1):
    df_diff = df.copy()
    for col in selected_cols:
        if order >= 1:
            df_diff[col] = df_diff[col].diff(order)
        if order >= 2:
            df_diff[col] = df_diff[col].diff(1)
    df_diff = df_diff.dropna(subset=selected_cols).reset_index(drop=True)
    return df_diff


def handle_missing_values(df, selected_cols, strategy="linear_interpolation"):
    df_clean = df.copy()
    if strategy == "linear_interpolation":
        df_clean[selected_cols] = df_clean[selected_cols].interpolate(method="linear")
        df_clean[selected_cols] = df_clean[selected_cols].bfill().ffill()
    elif strategy == "forward_fill":
        df_clean[selected_cols] = df_clean[selected_cols].ffill().bfill()
    elif strategy == "drop_rows":
        df_clean = df_clean.dropna(subset=selected_cols).reset_index(drop=True)
    return df_clean


def apply_standardization(df, selected_cols, method="zscore"):
    df_std = df.copy()
    if method == "zscore":
        for col in selected_cols:
            mean_val = df_std[col].mean()
            std_val = df_std[col].std()
            if std_val > 0:
                df_std[col] = (df_std[col] - mean_val) / std_val
    elif method == "minmax":
        for col in selected_cols:
            min_val = df_std[col].min()
            max_val = df_std[col].max()
            if max_val > min_val:
                df_std[col] = (df_std[col] - min_val) / (max_val - min_val)
    return df_std


def generate_adf_summary_plot(adf_df):
    fig, ax = plt.subplots(figsize=(10, max(3, len(adf_df) * 0.6)))
    colors = ["#16a34a" if s else "#dc2626" for s in adf_df["is_stationary"]]
    bars = ax.barh(adf_df["variable"], adf_df["adf_statistic"], color=colors, edgecolor="white", height=0.6)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("ADF Statistic", fontsize=10)
    ax.set_title("ADF Unit Root Test Results (Green=Stationary, Red=Non-stationary)", fontsize=11, fontweight="bold")
    for i, (val, pval) in enumerate(zip(adf_df["adf_statistic"], adf_df["p_value"])):
        label = f"stat={val:.3f}, p={pval:.4f}"
        ax.text(val, i, f"  {label}", va="center", fontsize=8,
                color=colors[i], fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()
    return fig
