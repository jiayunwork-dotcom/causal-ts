import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import networkx as nx


def plot_causal_graph(edges, selected_cols, title="Causal Directed Graph"):
    G = nx.DiGraph()
    for col in selected_cols:
        G.add_node(col)

    for edge in edges:
        G.add_edge(
            edge["source"], edge["target"],
            weight=edge["strength"],
            lag=edge.get("lag", 1),
            p_value=edge.get("p_value", np.nan)
        )

    if len(G.edges) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No significant causal edges found",
                ha="center", va="center", fontsize=14, color="gray")
        ax.set_title(title, fontsize=12, fontweight="bold")
        return fig

    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)

    degrees = dict(G.degree())
    node_sizes = [300 + degrees.get(n, 0) * 200 for n in G.nodes()]

    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    min_w = min(edge_weights) if edge_weights else 0
    edge_widths = [1 + 4 * (w - min_w) / (max_w - min_w + 1e-10) for w in edge_weights]

    edge_colors = []
    for u, v in G.edges():
        p = G[u][v].get("p_value", 0.05)
        if p < 0.01:
            edge_colors.append("#dc2626")
        elif p < 0.05:
            edge_colors.append("#f97316")
        else:
            edge_colors.append("#94a3b8")

    fig, ax = plt.subplots(figsize=(12, 9))

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color="#3b82f6", alpha=0.85, ax=ax,
                           edgecolors="white", linewidths=2)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold",
                            font_color="white", ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths,
                           edge_color=edge_colors, alpha=0.8,
                           arrows=True, arrowsize=20,
                           connectionstyle="arc3,rad=0.1",
                           min_source_margin=20, min_target_margin=20, ax=ax)

    edge_labels = {}
    for u, v in G.edges():
        lag = G[u][v].get("lag", "?")
        weight = G[u][v].get("weight", 0)
        edge_labels[(u, v)] = f"τ={lag}\n{weight:.3f}"

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=7, font_color="#1e293b",
                                 bbox=dict(boxstyle="round,pad=0.2",
                                           facecolor="white", alpha=0.7, edgecolor="none"),
                                 ax=ax)

    legend_elements = [
        mpatches.Patch(color="#dc2626", label="p < 0.01"),
        mpatches.Patch(color="#f97316", label="p < 0.05"),
        mpatches.Patch(color="#94a3b8", label="p ≥ 0.05"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              title="Significance", title_fontsize=9)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_lag_heatmap(granger_df, selected_cols, value_col="f_statistic", title="Lag Heatmap"):
    if granger_df is None or len(granger_df) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=14, color="gray")
        ax.set_title(title, fontsize=12, fontweight="bold")
        return fig

    n = len(selected_cols)
    matrix = np.zeros((n, n))
    col_idx = {col: i for i, col in enumerate(selected_cols)}

    for _, row in granger_df.iterrows():
        cause = row["cause"]
        effect = row["effect"]
        if cause in col_idx and effect in col_idx:
            val = row.get(value_col, 0)
            if not np.isnan(val):
                matrix[col_idx[effect], col_idx[cause]] = val

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n * 1.0)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(selected_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(selected_cols, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if val > 0:
                color = "white" if val > matrix.max() * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

    ax.set_xlabel("Cause", fontsize=10)
    ax.set_ylabel("Effect", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label=value_col)
    fig.tight_layout()
    return fig


def plot_causal_strength_matrix(granger_df, selected_cols, significance=0.05, title="Granger Causality Strength Matrix"):
    if granger_df is None or len(granger_df) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=14, color="gray")
        ax.set_title(title, fontsize=12, fontweight="bold")
        return fig

    n = len(selected_cols)
    f_matrix = np.zeros((n, n))
    sig_matrix = np.ones((n, n))
    col_idx = {col: i for i, col in enumerate(selected_cols)}

    for _, row in granger_df.iterrows():
        cause = row["cause"]
        effect = row["effect"]
        if cause in col_idx and effect in col_idx:
            f_val = row.get("f_statistic", 0)
            p_val = row.get("f_pvalue", 1.0)
            if not np.isnan(f_val):
                f_matrix[col_idx[effect], col_idx[cause]] = f_val
                sig_matrix[col_idx[effect], col_idx[cause]] = p_val

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n * 1.0)))
    im = ax.imshow(f_matrix, cmap="Blues", aspect="auto")

    for i in range(n):
        for j in range(n):
            val = f_matrix[i, j]
            p = sig_matrix[i, j]
            if val > 0:
                marker = "*" if p < significance else ""
                color = "white" if val > f_matrix.max() * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}{marker}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold" if p < significance else "normal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(selected_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(selected_cols, fontsize=9)
    ax.set_xlabel("Cause", fontsize=10)
    ax.set_ylabel("Effect", fontsize=10)
    ax.set_title(f"{title} (* = significant at p<{significance})", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="F-statistic")
    fig.tight_layout()
    return fig


def plot_lag_scatter(df, x_col, y_col, max_lag=10, title_prefix="Time Lag Scatter"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_lags = min(max_lag, len(df) // 3)
    correlations = []
    lags = range(0, n_lags + 1)

    for lag in lags:
        if lag == 0:
            corr = df[x_col].corr(df[y_col])
        else:
            corr = df[x_col].corr(df[y_col].shift(-lag))
        correlations.append(corr if not np.isnan(corr) else 0)

    axes[0].plot(list(lags), correlations, "o-", color="#2563eb", linewidth=1.5, markersize=5)
    axes[0].axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Lag", fontsize=10)
    axes[0].set_ylabel("Correlation", fontsize=10)
    axes[0].set_title(f"Cross-correlation: {x_col} → {y_col}", fontsize=10, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    best_lag = list(lags)[np.argmax(np.abs(correlations))]
    shift_data = pd.DataFrame({
        "x": df[x_col].values,
        "y": df[y_col].shift(-best_lag).values
    }).dropna()

    axes[1].scatter(shift_data["x"], shift_data["y"], alpha=0.4, s=15, color="#2563eb")
    if len(shift_data) > 2:
        z = np.polyfit(shift_data["x"], shift_data["y"], 1)
        p = np.poly1d(z)
        x_sorted = np.sort(shift_data["x"].values)
        axes[1].plot(x_sorted, p(x_sorted), "r--", linewidth=1.5, alpha=0.8)
    axes[1].set_xlabel(f"{x_col}", fontsize=10)
    axes[1].set_ylabel(f"{y_col} (lag={best_lag})", fontsize=10)
    axes[1].set_title(f"Scatter at optimal lag={best_lag} (r={correlations[best_lag]:.3f})",
                      fontsize=10, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"{title_prefix}: {x_col} → {y_col}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_transfer_entropy_heatmap(te_results, selected_cols, title="Transfer Entropy Heatmap"):
    n = len(selected_cols)
    matrix = np.zeros((n, n))
    col_idx = {col: i for i, col in enumerate(selected_cols)}

    for res in te_results:
        src = res["source"]
        tgt = res["target"]
        if src in col_idx and tgt in col_idx:
            matrix[col_idx[tgt], col_idx[src]] = res["transfer_entropy"]

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n * 1.0)))
    im = ax.imshow(matrix, cmap="Reds", aspect="auto")

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if val > 0:
                color = "white" if val > matrix.max() * 0.6 else "black"
                ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                        fontsize=8, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(selected_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(selected_cols, fontsize=9)
    ax.set_xlabel("Source", fontsize=10)
    ax.set_ylabel("Target", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Transfer Entropy")
    fig.tight_layout()
    return fig


def plot_var_roots(fitted_var, selected_cols, title="VAR Model Stability (Eigenvalue Roots)"):
    try:
        roots = fitted_var.roots
        moduli = np.abs(roots)
    except Exception:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.text(0.5, 0.5, "Could not compute VAR roots", ha="center", va="center",
                fontsize=14, color="gray")
        return fig

    fig, ax = plt.subplots(figsize=(7, 7))
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=1, alpha=0.5, label="Unit circle")

    for i, (root, modulus) in enumerate(zip(roots, moduli)):
        color = "#16a34a" if modulus < 1.0 else "#dc2626"
        marker = "o" if modulus < 1.0 else "x"
        ax.plot(root.real, root.imag, marker, color=color, markersize=8,
                markeredgewidth=2 if marker == "x" else 1)

    stable = all(m < 1.0 for m in moduli)
    status = "STABLE" if stable else "UNSTABLE"
    status_color = "#16a34a" if stable else "#dc2626"
    ax.set_title(f"{title}\nModel Status: {status}", fontsize=12, fontweight="bold",
                 color=status_color)
    ax.set_xlabel("Real", fontsize=10)
    ax.set_ylabel("Imaginary", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_anomaly_scatter(series, is_anomaly, var_name, title="Anomaly Detection"):
    """
    绘制单变量异常点散点图
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    n = len(series)
    indices = np.arange(n)

    normal_mask = ~is_anomaly
    anomaly_mask = is_anomaly

    ax.scatter(indices[normal_mask], series.values[normal_mask],
               c="#94a3b8", alpha=0.5, s=10, label="Normal")
    ax.scatter(indices[anomaly_mask], series.values[anomaly_mask],
               c="#ef4444", alpha=0.9, s=25, label="Anomaly", zorder=5)

    ax.set_xlabel("Time Index", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)
    ax.set_title(f"{title}: {var_name}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_all_anomaly_scatters(anomaly_results, selected_cols, method="consensus"):
    """
    绘制所有变量的异常散点图（子图）
    """
    from anomaly_detection import get_consensus_anomalies

    n_vars = len(selected_cols)
    nrows = min(n_vars, 4)
    ncols = (n_vars + nrows - 1) // nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    if n_vars == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, col in enumerate(selected_cols):
        if i >= len(axes):
            break
        var_res = anomaly_results["per_variable"][col]
        series = var_res["series"]
        is_anomaly = get_consensus_anomalies(anomaly_results, col, method)

        ax = axes[i]
        indices = np.arange(len(series))

        normal_mask = ~is_anomaly
        anomaly_mask = is_anomaly

        ax.scatter(indices[normal_mask], series.values[normal_mask],
                   c="#94a3b8", alpha=0.5, s=10, label="Normal")
        ax.scatter(indices[anomaly_mask], series.values[anomaly_mask],
                   c="#ef4444", alpha=0.9, s=25, label="Anomaly", zorder=5)

        ax.set_xlabel("Time Index", fontsize=9)
        ax.set_ylabel("Value", fontsize=9)
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(n_vars, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Anomaly Detection Results ({method.upper()})", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_root_cause_bar(root_cause_df, title="Root Cause Composite Score"):
    """
    绘制根因得分水平柱状图
    """
    if root_cause_df is None or len(root_cause_df) == 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No root cause data available",
                ha="center", va="center", fontsize=14, color="gray")
        ax.set_title(title, fontsize=12, fontweight="bold")
        return fig

    df = root_cause_df.copy()
    df = df.sort_values("composite_score", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.6)))

    cmap = plt.cm.RdBu_r
    norm = Normalize(vmin=df["composite_score"].min(), vmax=df["composite_score"].max())
    colors = [cmap(norm(score)) for score in df["composite_score"]]

    y_pos = range(len(df))
    bars = ax.barh(y_pos, df["composite_score"], color=colors, edgecolor="#333", linewidth=0.5)

    for i, (bar, score) in enumerate(zip(bars, df["composite_score"])):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["candidate_variable"], fontsize=10)
    ax.set_xlabel("Composite Score", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    return fig


def plot_anomaly_timeline(anomaly_results, selected_cols, target_col, root_cause_df,
                          method="consensus", title="Anomaly Propagation Timeline"):
    """
    绘制异常传播时间线图
    """
    from anomaly_detection import get_consensus_anomalies

    if root_cause_df is not None and len(root_cause_df) > 0:
        sorted_vars = root_cause_df["candidate_variable"].tolist()
        if target_col not in sorted_vars:
            sorted_vars = sorted_vars + [target_col]
        else:
            sorted_vars = [v for v in sorted_vars if v != target_col] + [target_col]
    else:
        sorted_vars = [v for v in selected_cols if v != target_col] + [target_col]

    n_vars = len(sorted_vars)
    if n_vars == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=14, color="gray")
        ax.set_title(title, fontsize=12, fontweight="bold")
        return fig

    n_time = len(anomaly_results["per_variable"][sorted_vars[0]]["series"])

    fig, ax = plt.subplots(figsize=(14, max(4, n_vars * 0.8)))

    y_pos = range(n_vars)

    for i, var in enumerate(sorted_vars):
        if var not in anomaly_results["per_variable"]:
            continue
        is_anomaly = get_consensus_anomalies(anomaly_results, var, method)
        anomaly_indices = np.where(is_anomaly)[0]

        ax.scatter(anomaly_indices, np.full_like(anomaly_indices, i, dtype=float),
                   c="#ef4444", s=40, zorder=5, edgecolors="white", linewidths=0.5)

    if root_cause_df is not None and len(root_cause_df) > 0:
        target_anomalies = get_consensus_anomalies(anomaly_results, target_col, method)
        target_anom_indices = np.where(target_anomalies)[0]

        target_y = n_vars - 1

        for i, var in enumerate(sorted_vars):
            if var == target_col:
                continue
            var_anomalies = get_consensus_anomalies(anomaly_results, var, method)
            var_anom_indices = np.where(var_anomalies)[0]

            if len(var_anom_indices) == 0 or len(target_anom_indices) == 0:
                continue

            for t_idx in target_anom_indices[:5]:
                best_c_idx = None
                best_lag = None
                for c_idx in var_anom_indices:
                    lag = t_idx - c_idx
                    if 0 <= lag <= 50:
                        if best_lag is None or lag < best_lag:
                            best_lag = lag
                            best_c_idx = c_idx
                if best_c_idx is not None:
                    ax.plot([best_c_idx, t_idx], [i, target_y],
                            color="#f97316", alpha=0.3, linewidth=1, linestyle="--", zorder=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_vars, fontsize=10)
    ax.set_xlabel("Time Index", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    if target_col in sorted_vars:
        target_idx = sorted_vars.index(target_col)
        ax.axhline(y=target_idx, color="#3b82f6", linestyle="--", alpha=0.7, linewidth=1, label="Target Variable")
        ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    return fig


def plot_scatter_compare(normal_df, abnormal_df, x_col, y_col, title_prefix=""):
    """
    正常段 vs 异常段散点图对比
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, df, label in [(axes[0], normal_df, "Normal Period"),
                           (axes[1], abnormal_df, "Abnormal Period")]:
        if df is None or len(df) < 2:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    fontsize=12, color="gray", transform=ax.transAxes)
            ax.set_title(f"{label}", fontsize=11, fontweight="bold")
            continue

        x = df[x_col].values
        y = df[y_col].values

        valid = ~np.isnan(x) & ~np.isnan(y)
        x = x[valid]
        y = y[valid]

        color = "#22c55e" if "Normal" in label else "#ef4444"

        ax.scatter(x, y, alpha=0.5, s=15, color=color)

        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_sorted = np.sort(x)
            ax.plot(x_sorted, p(x_sorted), "r--", linewidth=1.5, alpha=0.8)
            r = np.corrcoef(x, y)[0, 1]
            ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                    fontsize=10, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlabel(x_col, fontsize=10)
        ax.set_ylabel(y_col, fontsize=10)
        ax.set_title(f"{label}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{title_prefix} Scatter Comparison: {x_col} → {y_col}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_cross_correlation_compare(normal_df, abnormal_df, x_col, y_col, max_lag=20,
                                    title_prefix=""):
    """
    正常段 vs 异常段互相关函数对比
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = {"Normal": "#22c55e", "Abnormal": "#ef4444"}

    for df, label in [(normal_df, "Normal"), (abnormal_df, "Abnormal")]:
        if df is None or len(df) < max_lag + 5:
            continue

        x = df[x_col].values
        y = df[y_col].values

        valid = ~np.isnan(x) & ~np.isnan(y)
        x = x[valid]
        y = y[valid]

        if len(x) < max_lag + 2:
            continue

        lags = range(-max_lag, max_lag + 1)
        corrs = []
        for lag in lags:
            if lag >= 0:
                x_shifted = x[:len(x) - lag] if lag > 0 else x
                y_shifted = y[lag:]
            else:
                x_shifted = x[-lag:]
                y_shifted = y[:len(y) + lag]

            min_len = min(len(x_shifted), len(y_shifted))
            if min_len > 5:
                r = np.corrcoef(x_shifted[:min_len], y_shifted[:min_len])[0, 1]
                corrs.append(r if not np.isnan(r) else 0)
            else:
                corrs.append(0)

        ax.plot(list(lags), corrs, "o-", color=colors[label],
                linewidth=1.5, markersize=4, label=f"{label} Period")

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(x=0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel(f"Lag (positive = {x_col} leads)", fontsize=10)
    ax.set_ylabel("Cross-correlation", fontsize=10)
    ax.set_title(f"{title_prefix} Cross-correlation Comparison: {x_col} → {y_col}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 1)

    fig.tight_layout()
    return fig


def plot_propagation_graph(propagation_edges, selected_cols, root_cause, target_var,
                            strongest_path=None, title="Anomaly Propagation Graph"):
    """
    绘制异常传播子图的有向图
    
    节点颜色区分：根因=红色，目标=橙色，路径中间节点=蓝色，其他=灰色
    边粗细=因果强度，边上标注传播时滞
    最强路径用粗虚线高亮
    
    Parameters:
    -----------
    propagation_edges : list
        异常传播子图的边列表
    selected_cols : list
        所有选中的变量
    root_cause : str
        根因变量
    target_var : str
        目标变量
    strongest_path : dict or None
        最强路径（find_all_paths_bfs 返回的格式）
    title : str
        图标题
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    import networkx as nx
    
    G = nx.DiGraph()
    
    for col in selected_cols:
        G.add_node(col)
    
    for edge in propagation_edges:
        G.add_edge(
            edge["source"], edge["target"],
            weight=edge["strength"],
            lag=edge.get("lag", 1),
            anomaly_lag=edge.get("anomaly_lag", edge.get("lag", 1)),
            p_value=edge.get("p_value", np.nan)
        )
    
    path_nodes = set()
    path_edges_set = set()
    if strongest_path is not None:
        path_nodes = set(strongest_path["nodes"])
        for e in strongest_path["edges"]:
            path_edges_set.add((e["source"], e["target"]))
    
    def get_node_color(node):
        if node == root_cause:
            return "#ef4444"
        elif node == target_var:
            return "#f97316"
        elif node in path_nodes:
            return "#3b82f6"
        else:
            return "#9ca3af"
    
    node_colors = [get_node_color(n) for n in G.nodes()]
    
    if len(G.edges) == 0:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.text(0.5, 0.5, "No propagation edges found\nTry adjusting PCMCI parameters or anomaly detection sensitivity",
                ha="center", va="center", fontsize=14, color="gray", linespacing=1.5)
        ax.set_title(title, fontsize=13, fontweight="bold")
        return fig
    
    pos = nx.spring_layout(G, k=3.0, iterations=150, seed=42)
    
    degrees = dict(G.degree())
    node_sizes = [400 + degrees.get(n, 0) * 250 for n in G.nodes()]
    
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    min_w = min(edge_weights) if edge_weights else 0
    
    fig, ax = plt.subplots(figsize=(13, 9))
    
    for u, v in G.edges():
        is_path_edge = (u, v) in path_edges_set
        w = G[u][v]["weight"]
        if is_path_edge:
            width = 3.5 + 3 * (w - min_w) / (max_w - min_w + 1e-10)
        else:
            width = 1 + 2.5 * (w - min_w) / (max_w - min_w + 1e-10)
        
        edge_color = "#1e40af" if is_path_edge else "#64748b"
        style = "--" if is_path_edge else "solid"
        alpha = 0.95 if is_path_edge else 0.6
        
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            width=width,
            edge_color=edge_color,
            alpha=alpha,
            arrows=True,
            arrowsize=25 if is_path_edge else 18,
            arrowstyle="->",
            connectionstyle="arc3,rad=0.1",
            style=style,
            min_source_margin=25,
            min_target_margin=25,
            ax=ax
        )
    
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.9,
        ax=ax,
        edgecolors="white",
        linewidths=2.5
    )
    
    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_weight="bold",
        font_color="white",
        ax=ax
    )
    
    edge_labels = {}
    for u, v in G.edges():
        anomaly_lag = G[u][v].get("anomaly_lag", G[u][v].get("lag", "?"))
        weight = G[u][v].get("weight", 0)
        is_path_edge = (u, v) in path_edges_set
        marker = " ★" if is_path_edge else ""
        edge_labels[(u, v)] = f"τ={anomaly_lag}{marker}\n{weight:.3f}"
    
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=8,
        font_color="#1e293b",
        bbox=dict(boxstyle="round,pad=0.25",
                  facecolor="white", alpha=0.85, edgecolor="none"),
        ax=ax
    )
    
    legend_elements = [
        mpatches.Patch(color="#ef4444", label=f"Root Cause: {root_cause}"),
        mpatches.Patch(color="#f97316", label=f"Target: {target_var}"),
        mpatches.Patch(color="#3b82f6", label="Intermediate Node (on path)"),
        mpatches.Patch(color="#9ca3af", label="Other Node"),
    ]
    
    if strongest_path is not None:
        from matplotlib.lines import Line2D
        legend_elements.append(
            Line2D([0], [0], color="#1e40af", linewidth=3, linestyle="--", label="Strongest Path")
        )
    
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=9,
        title="Node / Edge Legend",
        title_fontsize=10,
        framealpha=0.95
    )
    
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.axis("off")
    fig.tight_layout()
    return fig

