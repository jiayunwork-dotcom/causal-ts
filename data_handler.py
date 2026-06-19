import io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_csv(file_path):
    df = pd.read_csv(file_path)
    if df.shape[1] < 2:
        return None, "CSV must have at least 2 columns (timestamp + 1 variable)"
    time_col = df.columns[0]
    try:
        df[time_col] = pd.to_datetime(df[time_col])
    except (ValueError, TypeError):
        try:
            df[time_col] = pd.to_numeric(df[time_col])
        except (ValueError, TypeError):
            return None, f"First column '{time_col}' must be a timestamp or numeric index"
    var_cols = df.columns[1:]
    for col in var_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(by=time_col).reset_index(drop=True)
    return df, None


def get_basic_statistics(df, selected_cols):
    stats = []
    for col in selected_cols:
        series = df[col]
        stats.append({
            "variable": col,
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing_count": series.isna().sum(),
            "missing_rate": series.isna().mean() * 100,
            "count": series.count()
        })
    return pd.DataFrame(stats)


def generate_time_series_preview(df, time_col, selected_cols, max_points=2000):
    n = len(df)
    if n > max_points:
        step = n // max_points
        plot_df = df.iloc[::step].copy()
    else:
        plot_df = df.copy()

    n_vars = len(selected_cols)
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 2.5 * n_vars), sharex=True)
    if n_vars == 1:
        axes = [axes]

    for ax, col in zip(axes, selected_cols):
        ax.plot(plot_df[time_col], plot_df[col], linewidth=0.8, color="#2563eb")
        ax.set_ylabel(col, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=8)

    axes[-1].set_xlabel("Time", fontsize=9)
    fig.suptitle("Time Series Preview", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def validate_column_selection(selected_cols):
    if len(selected_cols) < 2:
        return False, "Please select at least 2 columns for causal analysis"
    if len(selected_cols) > 10:
        return False, "Maximum 10 columns allowed for causal analysis"
    return True, None
