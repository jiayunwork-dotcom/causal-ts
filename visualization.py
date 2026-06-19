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
