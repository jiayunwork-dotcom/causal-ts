import os
import tempfile
import warnings
import uuid
from datetime import datetime
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from data_handler import parse_csv, get_basic_statistics, generate_time_series_preview, validate_column_selection
from preprocessing import (
    run_adf_tests, apply_differencing,
    handle_missing_values, apply_standardization, generate_adf_summary_plot
)
from granger_test import (
    bivariate_granger_test, multivariate_granger_test,
    conditional_granger_test
)
from transfer_entropy import pairwise_transfer_entropy
from pcmci import pcmci_algorithm
from visualization import (
    plot_causal_graph, plot_lag_heatmap, plot_causal_strength_matrix,
    plot_lag_scatter, plot_transfer_entropy_heatmap, plot_var_roots
)
from diagnostics import (
    run_full_diagnostics, apply_multiple_comparison_correction
)
from multiscale import multiscale_analysis, plot_multiscale_comparison, DEFAULT_SCALES
from report import generate_pdf_report
from anomaly_detection import (
    run_all_anomaly_detectors, get_consensus_anomalies, get_anomaly_indices
)
from root_cause import root_cause_analysis
from propagation_path import run_propagation_path_analysis
from visualization import (
    plot_all_anomaly_scatters, plot_root_cause_bar,
    plot_anomaly_timeline, plot_scatter_compare, plot_cross_correlation_compare,
    plot_propagation_graph
)

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class AnalysisState:
    def __init__(self):
        self.df = None
        self.time_col = None
        self.var_cols = []
        self.selected_cols = []
        self.processed_df = None
        self.adf_results = None
        self.granger_results = None
        self.multivar_results = None
        self.te_results = None
        self.pcmci_edges = None
        self.pcmci_link_matrix = None
        self.fitted_var = None
        self.lb_results = None
        self.stability_info = None
        self.corrected_granger = None
        self.ms_results = None
        self.snapshots = []
        self.last_preprocess_params = None
        self.anomaly_results = None
        self.anomaly_method = "consensus"
        self.root_cause_df = None
        self.root_cause_segments = None
        self.root_cause_target = None
        self.propagation_result = None
        self.propagation_root_cause = None
        self.propagation_target = None

    def reset(self):
        self.__init__()


state = AnalysisState()


def on_file_upload(file):
    if file is None:
        return (
            gr.update(choices=[], value=[]),
            "Please upload a CSV file",
            None
        )
    try:
        df, err = parse_csv(file.name)
        if err:
            return (
                gr.update(choices=[], value=[]),
                err,
                None
            )
        state.df = df
        state.time_col = df.columns[0]
        state.var_cols = df.columns[1:].tolist()

        stats_df = get_basic_statistics(df, state.var_cols)
        stats_str = stats_df.to_string(index=False)

        fig_preview = generate_time_series_preview(df, state.time_col, state.var_cols[:8])

        info = f"File loaded: {df.shape[0]} rows x {df.shape[1]} columns\n"
        info += f"Time column: {state.time_col}\n"
        info += f"Variables: {len(state.var_cols)}\n\n"
        info += f"Basic Statistics:\n{stats_str}"

        return (
            gr.update(choices=state.var_cols, value=state.var_cols[:min(5, len(state.var_cols))]),
            info,
            fig_preview
        )
    except Exception as e:
        return (
            gr.update(choices=[], value=[]),
            f"Error loading file: {str(e)}",
            None
        )


def on_column_select(selected):
    valid, err = validate_column_selection(selected)
    if not valid:
        return err, None, gr.update(choices=[]), gr.update(choices=[])
    state.selected_cols = list(selected)

    stats_df = get_basic_statistics(state.df, state.selected_cols)
    fig_preview = generate_time_series_preview(state.df, state.time_col, state.selected_cols)

    info = f"Selected {len(state.selected_cols)} variables: {', '.join(state.selected_cols)}\n\n"
    info += stats_df.to_string(index=False)

    return (
        info,
        fig_preview,
        gr.update(choices=list(state.selected_cols)),
        gr.update(choices=list(state.selected_cols))
    )


def on_preprocess(missing_strategy, standardize, diff_order, skip_stationarity):
    if state.df is None or not state.selected_cols:
        return "Please upload data and select columns first", None

    df = state.df.copy()

    df = handle_missing_values(df, state.selected_cols, missing_strategy)

    if standardize != "none":
        df = apply_standardization(df, state.selected_cols, standardize)

    adf_df, adf_results = run_adf_tests(df, state.selected_cols)

    warnings_list = []
    if skip_stationarity:
        warnings_list.append("WARNING: Stationarity check skipped - results may be unreliable")
        state.processed_df = df
    else:
        non_stationary = adf_df[adf_df["is_stationary"] == False]["variable"].tolist()
        if non_stationary and diff_order > 0:
            df = apply_differencing(df, state.selected_cols, diff_order)
            df = handle_missing_values(df, state.selected_cols, "linear_interpolation")
            adf_df, adf_results = run_adf_tests(df, state.selected_cols)

            still_non_stat = adf_df[adf_df["is_stationary"] == False]["variable"].tolist()
            if still_non_stat:
                warnings_list.append(
                    f"Variables still non-stationary after {diff_order}-order differencing: {', '.join(still_non_stat)}"
                )

        state.processed_df = df

    state.adf_results = adf_results
    state.last_preprocess_params = {
        "missing_strategy": missing_strategy,
        "standardize": standardize,
        "diff_order": diff_order,
        "skip_stationarity": skip_stationarity
    }

    fig_adf = generate_adf_summary_plot(adf_df)

    info = "Preprocessing complete\n\n"
    info += "ADF Test Results:\n"
    info += adf_df.to_string(index=False)
    if warnings_list:
        info += "\n\n" + "\n".join(warnings_list)

    return info, fig_adf


def on_bivariate_granger(max_lag, criterion, manual_lag):
    if state.processed_df is None:
        return "Please run preprocessing first", None, None, gr.update(choices=[])

    try:
        ml = int(manual_lag) if manual_lag and int(manual_lag) > 0 else None
        result_df = bivariate_granger_test(
            state.processed_df, state.selected_cols,
            int(max_lag), criterion, ml
        )
        state.granger_results = result_df

        sig_count = result_df["is_significant"].sum() if "is_significant" in result_df.columns else 0
        info = f"Bivariate Granger Test Results\n"
        info += f"Significant pairs: {sig_count}/{len(result_df)}\n\n"
        info += result_df.to_string(index=False)

        fig_strength = plot_causal_strength_matrix(result_df, state.selected_cols)
        fig_heatmap = plot_lag_heatmap(result_df, state.selected_cols)

        pair_options = []
        for _, row in result_df.iterrows():
            pair_options.append(f"{row['cause']} -> {row['effect']}")

        return (
            info,
            fig_strength,
            fig_heatmap,
            gr.update(choices=pair_options, value=pair_options[0] if pair_options else None)
        )
    except Exception as e:
        return f"Error in Granger test: {str(e)}", None, None, gr.update(choices=[])


def on_multivariate_granger(max_lag, criterion):
    if state.processed_df is None:
        return "Please run preprocessing first", None

    try:
        result_df, err = multivariate_granger_test(
            state.processed_df, state.selected_cols,
            int(max_lag), criterion
        )
        if err:
            return err, None

        state.multivar_results = result_df

        info = "Multivariate Granger Test (VAR Wald Test)\n\n"
        info += result_df.to_string(index=False)

        fig_heatmap = plot_lag_heatmap(result_df, state.selected_cols, value_col="wald_statistic",
                                        title="Multivariate Wald Test Heatmap")

        return info, fig_heatmap
    except Exception as e:
        return f"Error: {str(e)}", None


def on_conditional_granger(cause, effect, control, max_lag, criterion):
    if state.processed_df is None:
        return "Please run preprocessing first"

    if not cause or not effect:
        return "Please select cause and effect variables"

    control_cols = [c.strip() for c in control.split(",") if c.strip()] if control else []
    for c in control_cols:
        if c not in state.selected_cols:
            return f"Control variable '{c}' not in selected columns"

    try:
        result, err = conditional_granger_test(
            state.processed_df, cause, effect, control_cols,
            int(max_lag), criterion
        )
        if err:
            return err

        info = "Conditional Granger Causality Test\n"
        info += f"Cause: {result['cause']}\n"
        info += f"Effect: {result['effect']}\n"
        info += f"Controlling for: {result['controlling_for']}\n"
        info += f"Wald statistic: {result['wald_statistic']}\n"
        info += f"P-value: {result['wald_pvalue']}\n"
        info += f"Significant: {result['is_significant']}\n"
        info += f"Optimal lag: {result['optimal_lag']}"

        return info
    except Exception as e:
        return f"Error: {str(e)}"


def on_transfer_entropy(embedding_dim, k_neighbors, n_surrogates):
    if state.processed_df is None:
        return "Please run preprocessing first", None

    try:
        results = pairwise_transfer_entropy(
            state.processed_df, state.selected_cols,
            embedding_dim=int(embedding_dim),
            k=int(k_neighbors),
            n_surrogates=int(n_surrogates)
        )
        state.te_results = results

        te_df = pd.DataFrame(results)
        info = "Transfer Entropy Results\n\n"
        info += te_df.to_string(index=False)

        fig_te = plot_transfer_entropy_heatmap(results, state.selected_cols)

        return info, fig_te
    except Exception as e:
        return f"Error: {str(e)}", None


def on_pcmci(tau_max, alpha, ci_test):
    if state.processed_df is None:
        return "Please run preprocessing first", None, None

    try:
        edges, link_matrix, pc_parents, mci_results = pcmci_algorithm(
            state.processed_df, state.selected_cols,
            tau_max=int(tau_max),
            alpha=float(alpha),
            ci_test=ci_test
        )
        state.pcmci_edges = edges
        state.pcmci_link_matrix = link_matrix

        info = f"PCMCI Causal Discovery Results\n"
        info += f"Significant causal edges: {len(edges)}\n\n"
        if edges:
            edges_df = pd.DataFrame(edges)
            info += edges_df.to_string(index=False)
        else:
            info += "No significant causal edges found"

        fig_graph = plot_causal_graph(edges, state.selected_cols, title="PCMCI Causal Directed Graph")

        pcmci_edges_for_heatmap = []
        for e in edges:
            pcmci_edges_for_heatmap.append({
                "cause": e["source"],
                "effect": e["target"],
                "f_statistic": e["strength"],
                "f_pvalue": e.get("p_value", np.nan),
                "is_significant": True
            })
        fig_heatmap = None
        if pcmci_edges_for_heatmap:
            fig_heatmap = plot_lag_heatmap(
                pd.DataFrame(pcmci_edges_for_heatmap), state.selected_cols,
                value_col="f_statistic", title="PCMCI Causal Strength Heatmap"
            )

        return info, fig_graph, fig_heatmap
    except Exception as e:
        return f"Error: {str(e)}", None, None


def on_lag_scatter(pair_selection, max_lag):
    if state.processed_df is None or not pair_selection:
        return None

    try:
        parts = pair_selection.split("->")
        if len(parts) != 2:
            return None
        x_col = parts[0].strip()
        y_col = parts[1].strip()

        fig = plot_lag_scatter(
            state.processed_df, x_col, y_col, int(max_lag)
        )
        return fig
    except Exception as e:
        print(f"Scatter plot error: {e}")
        return None


def on_diagnostics(max_lag, criterion):
    if state.processed_df is None:
        return "Please run preprocessing first", None

    try:
        fitted_var, lb_df, stability_info, err = run_full_diagnostics(
            state.processed_df, state.selected_cols, int(max_lag), criterion
        )
        if err:
            return err, None

        state.fitted_var = fitted_var
        state.lb_results = lb_df
        state.stability_info = stability_info

        info = "Model Diagnostics\n\n"
        info += "Ljung-Box Residual Test:\n"
        info += lb_df.to_string(index=False)
        info += "\n\n"

        if stability_info:
            info += f"VAR Stability: {'STABLE' if stability_info['is_stable'] else 'UNSTABLE'}\n"
            info += f"Max eigenvalue modulus: {stability_info.get('max_modulus', 'N/A')}\n"
            if stability_info.get('n_roots_outside', 0) > 0:
                info += f"Roots outside unit circle: {stability_info['n_roots_outside']}"

        fig_roots = None
        if fitted_var:
            fig_roots = plot_var_roots(fitted_var, state.selected_cols)

        return info, fig_roots
    except Exception as e:
        return f"Error: {str(e)}", None


def on_correction(method, alpha):
    if state.granger_results is None:
        return "Please run Granger test first"

    try:
        corrected = apply_multiple_comparison_correction(
            state.granger_results, method, float(alpha)
        )
        state.corrected_granger = corrected

        sig_col = f"significant_{method}"
        sig_count = corrected[sig_col].sum() if sig_col in corrected.columns else 0

        info = f"Multiple Comparison Correction: {method.upper()}\n"
        info += f"Significant after correction: {sig_count}/{len(corrected)}\n\n"
        info += corrected.to_string(index=False)

        return info
    except Exception as e:
        return f"Error: {str(e)}"


def on_multiscale(scales_str, max_lag, criterion):
    if state.processed_df is None:
        return "Please run preprocessing first", None

    try:
        scales = {}
        for item in scales_str.split(","):
            parts = item.strip().split(":")
            if len(parts) == 2:
                scales[parts[0].strip()] = parts[1].strip()

        if not scales:
            scales = DEFAULT_SCALES

        results = multiscale_analysis(
            state.processed_df, state.time_col,
            state.selected_cols, scales, int(max_lag), criterion
        )
        state.ms_results = results

        info = "Multi-scale Analysis Results\n\n"
        for scale_name, res in results.items():
            if res["error"]:
                info += f"[{scale_name}] Error: {res['error']}\n"
            elif res["granger"] is not None:
                sig = res["granger"]["is_significant"].sum() if "is_significant" in res["granger"].columns else 0
                info += f"[{scale_name}] n={res['n_obs']}, significant pairs: {sig}/{len(res['granger'])}\n"
        info += "\nSee visualization for comparison"

        fig = plot_multiscale_comparison(results, state.selected_cols)

        return info, fig
    except Exception as e:
        return f"Error: {str(e)}", None


def on_generate_report():
    try:
        fig_paths = {}

        if state.processed_df is not None and state.selected_cols:
            fig_preview = generate_time_series_preview(
                state.processed_df, state.time_col, state.selected_cols
            )
            preview_path = os.path.join(tempfile.gettempdir(), "preview.png")
            fig_preview.savefig(preview_path, dpi=120, bbox_inches="tight")
            plt.close(fig_preview)
            fig_paths["preview"] = preview_path

        if state.granger_results is not None:
            fig_strength = plot_causal_strength_matrix(
                state.granger_results, state.selected_cols
            )
            strength_path = os.path.join(tempfile.gettempdir(), "strength_matrix.png")
            fig_strength.savefig(strength_path, dpi=120, bbox_inches="tight")
            plt.close(fig_strength)
            fig_paths["strength_matrix"] = strength_path

        if state.pcmci_edges is not None:
            fig_graph = plot_causal_graph(state.pcmci_edges, state.selected_cols)
            graph_path = os.path.join(tempfile.gettempdir(), "causal_graph.png")
            fig_graph.savefig(graph_path, dpi=120, bbox_inches="tight")
            plt.close(fig_graph)
            fig_paths["causal_graph"] = graph_path

        adf_df = None
        if state.adf_results:
            adf_df = pd.DataFrame(state.adf_results)

        stats_df = get_basic_statistics(
            state.processed_df if state.processed_df is not None else state.df,
            state.selected_cols
        ) if state.df is not None else None

        report_path = generate_pdf_report(
            stats_df=stats_df,
            adf_df=adf_df,
            granger_df=state.granger_results,
            multivar_df=state.multivar_results,
            te_results=state.te_results,
            pcmci_edges=state.pcmci_edges,
            lb_df=state.lb_results,
            stability_info=state.stability_info,
            selected_cols=state.selected_cols,
            figures=fig_paths
        )

        return report_path
    except Exception as e:
        return f"Error generating report: {str(e)}"


def on_run_anomaly_detection(zscore_window, zscore_threshold, cusum_k, cusum_h, if_contamination):
    if state.processed_df is None or not state.selected_cols:
        return "Please run preprocessing and select columns first", None, None, gr.update(choices=[]), gr.update(choices=[])

    try:
        anomaly_results = run_all_anomaly_detectors(
            state.processed_df, state.selected_cols,
            zscore_window=int(zscore_window),
            zscore_threshold=float(zscore_threshold),
            cusum_k=float(cusum_k),
            cusum_h=float(cusum_h),
            if_contamination=float(if_contamination)
        )
        state.anomaly_results = anomaly_results
        state.anomaly_method = "consensus"

        info = "Anomaly Detection Results Summary\n\n"
        info += anomaly_results["summary"].to_string(index=False)

        scatter_fig = plot_all_anomaly_scatters(anomaly_results, state.selected_cols, "consensus")

        var_choices = list(state.selected_cols)
        default_target = var_choices[0] if var_choices else None

        return info, scatter_fig, anomaly_results["summary"], \
               gr.update(choices=var_choices, value=default_target), \
               gr.update(choices=var_choices, value=default_target)
    except Exception as e:
        return f"Error in anomaly detection: {str(e)}", None, None, gr.update(choices=[]), gr.update(choices=[])


def on_change_anomaly_method(method):
    if state.anomaly_results is None:
        return None

    state.anomaly_method = method
    scatter_fig = plot_all_anomaly_scatters(state.anomaly_results, state.selected_cols, method)
    return scatter_fig


def on_run_root_cause(target_var, window_size, max_lag, criterion):
    if state.anomaly_results is None:
        return "Please run anomaly detection first", None, None, None, None

    if not target_var or target_var not in state.selected_cols:
        return "Please select a target variable", None, None, None, None

    try:
        candidate_cols = [c for c in state.selected_cols if c != target_var]

        if len(candidate_cols) == 0:
            return "Need at least 2 variables for root cause analysis", None, None, None, None

        root_cause_df, segments = root_cause_analysis(
            state.processed_df, target_var, candidate_cols,
            state.anomaly_results,
            anomaly_method=state.anomaly_method,
            window_size=int(window_size),
            max_lag=int(max_lag),
            criterion=criterion
        )

        state.root_cause_df = root_cause_df
        state.root_cause_segments = segments
        state.root_cause_target = target_var

        if root_cause_df is None or len(root_cause_df) == 0:
            return "No anomalies detected for root cause analysis", None, None, None, None

        info = "Root Cause Analysis Results\n\n"
        display_cols = ["candidate_variable", "normal_f_value", "abnormal_f_value",
                        "change_rate", "propagation_lag", "composite_score"]
        display_df = root_cause_df[display_cols].copy()
        display_df.columns = ["Candidate", "Normal F", "Abnormal F", "Change Rate", "Prop Lag", "Score"]
        info += display_df.to_string(index=False)

        bar_fig = plot_root_cause_bar(root_cause_df)
        timeline_fig = plot_anomaly_timeline(
            state.anomaly_results, state.selected_cols,
            target_var, root_cause_df, state.anomaly_method
        )

        candidates = root_cause_df["candidate_variable"].tolist()
        default_candidate = candidates[0] if candidates else None

        return info, bar_fig, timeline_fig, gr.update(choices=candidates, value=default_candidate), None
    except Exception as e:
        return f"Error in root cause analysis: {str(e)}", None, None, None, None


def on_select_candidate_verification(candidate_var):
    if state.root_cause_df is None or state.root_cause_segments is None:
        return None, None

    if not candidate_var or state.root_cause_target is None:
        return None, None

    try:
        normal_df = state.root_cause_segments["normal_df"]
        abnormal_df = state.root_cause_segments["abnormal_df"]
        target_var = state.root_cause_target

        scatter_fig = plot_scatter_compare(
            normal_df, abnormal_df, candidate_var, target_var,
            title_prefix="Root Cause"
        )

        ccf_fig = plot_cross_correlation_compare(
            normal_df, abnormal_df, candidate_var, target_var,
            max_lag=20, title_prefix="Root Cause"
        )

        return scatter_fig, ccf_fig
    except Exception as e:
        print(f"Verification plot error: {e}")
        return None, None


def on_infer_propagation_path(pp_tau_max, pp_alpha, pp_ci_test):
    if state.root_cause_df is None or len(state.root_cause_df) == 0:
        return "Please run Root Cause Analysis first", None, None, pd.DataFrame(
            columns=["Rank", "Path", "Total Lag", "Avg Causal Strength"]
        )

    if state.anomaly_results is None or state.processed_df is None:
        return "Please run anomaly detection and preprocessing first", None, None, pd.DataFrame(
            columns=["Rank", "Path", "Total Lag", "Avg Causal Strength"]
        )

    target_var = state.root_cause_target
    if not target_var:
        return "No target variable set", None, None, pd.DataFrame(
            columns=["Rank", "Path", "Total Lag", "Avg Causal Strength"]
        )

    root_cause_var = state.root_cause_df.iloc[0]["candidate_variable"]

    try:
        pcmci_params = {
            "tau_max": int(pp_tau_max),
            "alpha": float(pp_alpha),
            "ci_test": pp_ci_test
        }

        result = run_propagation_path_analysis(
            state.processed_df,
            state.selected_cols,
            root_cause_var,
            target_var,
            state.anomaly_results,
            pcmci_params,
            anomaly_method=state.anomaly_method,
            top_k=3
        )

        state.propagation_result = result
        state.propagation_root_cause = root_cause_var
        state.propagation_target = target_var

        propagation_edges = result["propagation_edges"]
        strongest_path = result["strongest_path"]
        all_paths = result["all_paths"]
        top_paths = result["top_paths"]

        graph_fig = plot_propagation_graph(
            propagation_edges,
            state.selected_cols,
            root_cause_var,
            target_var,
            strongest_path=strongest_path,
            title=f"Anomaly Propagation Graph: {root_cause_var} → {target_var}"
        )

        info = f"Propagation Path Analysis Complete\n"
        info += f"Root Cause: {root_cause_var}\n"
        info += f"Target Variable: {target_var}\n"
        info += f"PCMCI causal edges: {len(result['pcmci_edges'])}\n"
        info += f"Propagation edges: {len(propagation_edges)}\n"
        info += f"Paths found: {len(all_paths)}\n\n"

        if strongest_path is not None:
            nodes = strongest_path["nodes"]
            path_str_parts = []
            for i, node in enumerate(nodes):
                if i == 0:
                    path_str_parts.append(node)
                else:
                    edge_info = strongest_path["edges"][i - 1]
                    path_str_parts.append(f"{node} (τ={edge_info['anomaly_lag']})")
            path_str = " → ".join(path_str_parts)

            info += "=== Inferred Propagation Path (Strongest) ===\n"
            info += f"{path_str}\n"
            info += f"Total Lag: {strongest_path['total_lag']}\n"
            info += f"Avg Causal Strength: {strongest_path['avg_strength']}\n\n"

            if len(top_paths) > 1:
                info += f"=== Other Strong Paths (Top {len(top_paths)}) ===\n"
                for pi, p in enumerate(top_paths[1:], start=2):
                    n = p["nodes"]
                    ps_parts = []
                    for i, node in enumerate(n):
                        if i == 0:
                            ps_parts.append(node)
                        else:
                            ei = p["edges"][i - 1]
                            ps_parts.append(f"{node} (τ={ei['anomaly_lag']})")
                    ps = " → ".join(ps_parts)
                    info += f"Path #{pi}: {ps} | Avg Strength={p['avg_strength']}, Total Lag={p['total_lag']}\n"
                info += "\n"

            summary = _generate_path_summary(strongest_path)
            info += f"Summary Text:\n{summary}"
        else:
            msg = "未发现从根因到目标的连通路径,建议检查因果图参数或异常检测灵敏度"
            info += f"No connected path found from root cause to target.\n{msg}"

        summary_text_raw = _generate_path_summary(strongest_path)
        if strongest_path is not None:
            summary_md = f"""
            <div style="background-color: #f0f9ff; border-left: 4px solid #3b82f6; padding: 15px; border-radius: 6px; margin: 10px 0;">
                <p style="font-size: 15px; font-weight: 600; margin: 0; color: #1e40af;">📊 {summary_text_raw}</p>
            </div>
            """
        else:
            summary_md = f"""
            <div style="background-color: #fef2f2; border-left: 4px solid #ef4444; padding: 15px; border-radius: 6px; margin: 10px 0;">
                <p style="font-size: 15px; font-weight: 600; margin: 0; color: #991b1b;">⚠️ {summary_text_raw}</p>
            </div>
            """

        paths_df_data = []
        for pi, p in enumerate(top_paths, start=1):
            n = p["nodes"]
            ps_parts = []
            for i, node in enumerate(n):
                if i == 0:
                    ps_parts.append(node)
                else:
                    ei = p["edges"][i - 1]
                    ps_parts.append(f"{node}(τ={ei['anomaly_lag']})")
            ps_str = " → ".join(ps_parts)
            paths_df_data.append({
                "Rank": pi,
                "Path": ps_str,
                "Total Lag": p["total_lag"],
                "Avg Causal Strength": p["avg_strength"]
            })

        paths_df = pd.DataFrame(paths_df_data) if paths_df_data else pd.DataFrame(
            columns=["Rank", "Path", "Total Lag", "Avg Causal Strength"]
        )

        return info, graph_fig, gr.Markdown(summary_md), paths_df

    except Exception as e:
        import traceback
        traceback.print_exc()
        err_md = f"""
        <div style="background-color: #fef2f2; border-left: 4px solid #ef4444; padding: 15px; border-radius: 6px; margin: 10px 0;">
            <p style="font-size: 15px; font-weight: 600; margin: 0; color: #991b1b;">⚠️ Error: {str(e)}</p>
        </div>
        """
        return (f"Error in propagation path inference: {str(e)}", None,
                gr.Markdown(err_md),
                pd.DataFrame(columns=["Rank", "Path", "Total Lag", "Avg Causal Strength"]))


def _generate_path_summary(strongest_path):
    if strongest_path is None:
        return "未发现从根因到目标的连通路径,建议检查因果图参数或异常检测灵敏度"

    nodes = strongest_path["nodes"]
    edges = strongest_path["edges"]
    total_lag = strongest_path["total_lag"]
    avg_strength = strongest_path["avg_strength"]

    if len(nodes) < 2:
        return "未发现从根因到目标的连通路径,建议检查因果图参数或异常检测灵敏度"

    parts = []
    for i, node in enumerate(nodes):
        if i == 0:
            parts.append(node)
        else:
            lag_val = edges[i - 1]["anomaly_lag"]
            parts.append(f"{node} (时滞={lag_val})")

    path_str = " → ".join(parts)
    summary = f"推断传播路径: {path_str},路径总时滞={total_lag},平均因果强度={avg_strength}"
    return summary


def _method_display_name(method_key):
    names = {
        "bivariate_granger": "Bivariate Granger",
        "multivariate_granger": "Multivariate Granger",
        "pcmci": "PCMCI",
        "transfer_entropy": "Transfer Entropy"
    }
    return names.get(method_key, method_key)


def _flatten_params(snapshot):
    flat = {}
    flat["Variables"] = ", ".join(snapshot["variables"])
    flat["Missing Strategy"] = snapshot["preprocessing"]["missing_strategy"]
    flat["Standardization"] = snapshot["preprocessing"]["standardize"]
    flat["Differencing Order"] = str(snapshot["preprocessing"]["diff_order"])
    flat["Skip Stationarity"] = str(snapshot["preprocessing"]["skip_stationarity"])

    params = snapshot["parameters"]
    if snapshot["method"] == "bivariate_granger":
        flat["Max Lag"] = str(params.get("max_lag", ""))
        flat["Criterion"] = params.get("criterion", "")
        flat["Manual Lag"] = str(params.get("manual_lag", "auto"))
    elif snapshot["method"] == "multivariate_granger":
        flat["Max Lag"] = str(params.get("max_lag", ""))
        flat["Criterion"] = params.get("criterion", "")
    elif snapshot["method"] == "pcmci":
        flat["Tau Max"] = str(params.get("tau_max", ""))
        flat["Alpha"] = str(params.get("alpha", ""))
        flat["CI Test"] = params.get("ci_test", "")
    elif snapshot["method"] == "transfer_entropy":
        flat["Embedding Dim"] = str(params.get("embedding_dim", ""))
        flat["K Neighbors"] = str(params.get("k_neighbors", ""))
        flat["N Surrogates"] = str(params.get("n_surrogates", ""))

    return flat


def _create_snapshot(label, method, parameters):
    snapshot = {
        "id": str(uuid.uuid4())[:8],
        "label": label,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": method,
        "variables": list(state.selected_cols),
        "preprocessing": dict(state.last_preprocess_params) if state.last_preprocess_params else {},
        "parameters": parameters,
        "results": []
    }
    return snapshot


def save_snapshot_bivariate(label, max_lag, criterion, manual_lag):
    if not label or not label.strip():
        return "Please enter a snapshot label", gr.update(choices=[]), gr.update(choices=[])
    label = label.strip()

    if state.granger_results is None:
        return "Run the test first before saving a snapshot.", gr.update(choices=[]), gr.update(choices=[])

    params = {
        "max_lag": int(max_lag),
        "criterion": criterion,
        "manual_lag": int(manual_lag) if manual_lag and int(manual_lag) > 0 else "auto"
    }
    snapshot = _create_snapshot(label, "bivariate_granger", params)

    for _, row in state.granger_results.iterrows():
        snapshot["results"].append({
            "cause": row["cause"],
            "effect": row["effect"],
            "statistic": float(row["f_statistic"]) if pd.notna(row["f_statistic"]) else None,
            "p_value": float(row["f_pvalue"]) if pd.notna(row["f_pvalue"]) else None,
            "is_significant": bool(row["is_significant"]) if row["is_significant"] is not None else None
        })

    state.snapshots.append(snapshot)
    choices = _get_snapshot_choices()
    pair_options = _get_all_pair_options()
    return (f"Snapshot '{label}' saved! ({len(state.snapshots)} total)",
            gr.update(choices=choices, value=[]),
            gr.update(choices=pair_options, value=None))


def save_snapshot_multivariate(label, max_lag, criterion):
    if not label or not label.strip():
        return "Please enter a snapshot label", gr.update(choices=[]), gr.update(choices=[])
    label = label.strip()

    if state.multivar_results is None:
        return "Run the test first before saving a snapshot.", gr.update(choices=[]), gr.update(choices=[])

    params = {
        "max_lag": int(max_lag),
        "criterion": criterion
    }
    snapshot = _create_snapshot(label, "multivariate_granger", params)

    for _, row in state.multivar_results.iterrows():
        snapshot["results"].append({
            "cause": row["cause"],
            "effect": row["effect"],
            "statistic": float(row["wald_statistic"]) if pd.notna(row["wald_statistic"]) else None,
            "p_value": float(row["wald_pvalue"]) if pd.notna(row["wald_pvalue"]) else None,
            "is_significant": bool(row["is_significant"]) if row["is_significant"] is not None else None
        })

    state.snapshots.append(snapshot)
    choices = _get_snapshot_choices()
    pair_options = _get_all_pair_options()
    return (f"Snapshot '{label}' saved! ({len(state.snapshots)} total)",
            gr.update(choices=choices, value=[]),
            gr.update(choices=pair_options, value=None))


def save_snapshot_pcmci(label, tau_max, alpha, ci_test):
    if not label or not label.strip():
        return "Please enter a snapshot label", gr.update(choices=[]), gr.update(choices=[])
    label = label.strip()

    if state.pcmci_edges is None:
        return "Run the algorithm first before saving a snapshot.", gr.update(choices=[]), gr.update(choices=[])

    params = {
        "tau_max": int(tau_max),
        "alpha": float(alpha),
        "ci_test": ci_test
    }
    snapshot = _create_snapshot(label, "pcmci", params)

    all_pairs = []
    for v1 in state.selected_cols:
        for v2 in state.selected_cols:
            if v1 != v2:
                all_pairs.append((v1, v2))
    edge_keys = set()
    for e in state.pcmci_edges:
        edge_keys.add((e["source"], e["target"]))

    for cause, effect in all_pairs:
        if (cause, effect) in edge_keys:
            e = next(e for e in state.pcmci_edges if e["source"] == cause and e["target"] == effect)
            snapshot["results"].append({
                "cause": cause,
                "effect": effect,
                "statistic": float(e.get("strength")) if e.get("strength") is not None else None,
                "p_value": float(e.get("p_value")) if e.get("p_value") is not None else None,
                "is_significant": True
            })
        else:
            snapshot["results"].append({
                "cause": cause,
                "effect": effect,
                "statistic": None,
                "p_value": None,
                "is_significant": False
            })

    state.snapshots.append(snapshot)
    choices = _get_snapshot_choices()
    pair_options = _get_all_pair_options()
    return (f"Snapshot '{label}' saved! ({len(state.snapshots)} total)",
            gr.update(choices=choices, value=[]),
            gr.update(choices=pair_options, value=None))


def save_snapshot_transfer_entropy(label, embedding_dim, k_neighbors, n_surrogates):
    if not label or not label.strip():
        return "Please enter a snapshot label", gr.update(choices=[]), gr.update(choices=[])
    label = label.strip()

    if state.te_results is None:
        return "Run the analysis first before saving a snapshot.", gr.update(choices=[]), gr.update(choices=[])

    params = {
        "embedding_dim": int(embedding_dim),
        "k_neighbors": int(k_neighbors),
        "n_surrogates": int(n_surrogates)
    }
    snapshot = _create_snapshot(label, "transfer_entropy", params)

    for r in state.te_results:
        snapshot["results"].append({
            "cause": r["source"],
            "effect": r["target"],
            "statistic": float(r["transfer_entropy"]),
            "p_value": float(r["p_value"]),
            "is_significant": bool(r["is_significant"])
        })

    state.snapshots.append(snapshot)
    choices = _get_snapshot_choices()
    pair_options = _get_all_pair_options()
    return (f"Snapshot '{label}' saved! ({len(state.snapshots)} total)",
            gr.update(choices=choices, value=[]),
            gr.update(choices=pair_options, value=None))


def _get_snapshot_choices():
    choices = []
    for i, s in enumerate(state.snapshots):
        display = f"{s['label']} [{_method_display_name(s['method'])}] - {s['timestamp']}"
        choices.append((display, i))
    return choices


def _get_all_pair_options():
    all_pairs = set()
    for s in state.snapshots:
        for r in s["results"]:
            pair_key = f"{r['cause']} -> {r['effect']}"
            all_pairs.add(pair_key)
    return sorted(list(all_pairs))


def _indices_from_selected(selected_values):
    if not selected_values:
        return []
    indices = []
    for v in selected_values:
        try:
            indices.append(int(v))
        except (ValueError, TypeError):
            pass
    return sorted(indices)


def get_snapshot_list():
    choices = _get_snapshot_choices()
    return gr.update(choices=choices, value=[])


def delete_snapshot(selected_values):
    indices = _indices_from_selected(selected_values)
    if not indices:
        choices = _get_snapshot_choices()
        return "No snapshot selected for deletion", gr.update(choices=choices, value=[]), gr.update(choices=[], value=None), gr.HTML("<p style='color: #888;'>Select 2-4 snapshots to compare parameters</p>"), None, None

    indices_to_remove = sorted(indices, reverse=True)
    deleted_labels = []
    for idx in indices_to_remove:
        if 0 <= idx < len(state.snapshots):
            deleted_labels.append(state.snapshots[idx]["label"])
            del state.snapshots[idx]

    choices = _get_snapshot_choices()
    pair_options = _get_all_pair_options()

    return (f"Deleted snapshot(s): {', '.join(deleted_labels)}",
            gr.update(choices=choices, value=[]),
            gr.update(choices=pair_options, value=None),
            gr.HTML("<p style='color: #888;'>Select 2-4 snapshots to compare parameters</p>"),
            None, None)


def clear_all_snapshots():
    count = len(state.snapshots)
    state.snapshots.clear()
    return (f"Cleared {count} snapshot(s)",
            gr.update(choices=[], value=[]),
            gr.update(choices=[], value=None),
            gr.HTML("<p style='color: #888;'>Select 2-4 snapshots to compare parameters</p>"),
            None, None)


def get_param_diff_html(selected_indices):
    if not selected_indices or len(selected_indices) < 2:
        return "<p style='color: #888;'>Please select 2-4 snapshots to compare</p>"

    selected_snapshots = [state.snapshots[i] for i in selected_indices if i < len(state.snapshots)]
    if len(selected_snapshots) < 2:
        return "<p style='color: #888;'>Please select 2-4 snapshots to compare</p>"

    all_params = [_flatten_params(s) for s in selected_snapshots]
    param_keys = set()
    for p in all_params:
        param_keys.update(p.keys())
    param_keys = sorted(param_keys)

    html = "<div style='overflow-x: auto;'>"
    html += "<table style='width: 100%; border-collapse: collapse; font-size: 13px;'>"
    html += "<thead><tr><th style='padding: 8px; border: 1px solid #ddd; background: #f0f0f0; text-align: left;'>Parameter</th>"
    for s in selected_snapshots:
        html += f"<th style='padding: 8px; border: 1px solid #ddd; background: #f0f0f0; text-align: center;'>{s['label']}</th>"
    html += "</tr></thead><tbody>"

    for key in param_keys:
        values = [p.get(key, "-") for p in all_params]
        all_same = len(set(values)) == 1
        bg_style = "background: #f5f5f5; color: #888;" if all_same else "background: #fff9c4;"

        html += f"<tr><td style='padding: 8px; border: 1px solid #ddd; font-weight: 500;'>{key}</td>"
        for v in values:
            html += f"<td style='padding: 8px; border: 1px solid #ddd; text-align: center; {bg_style}'>{v}</td>"
        html += "</tr>"

    html += "</tbody></table></div>"
    html += "<p style='font-size: 12px; color: #666; margin-top: 8px;'><span style='display: inline-block; width: 12px; height: 12px; background: #f5f5f5; border: 1px solid #ddd; margin-right: 4px;'></span> Same value &nbsp;&nbsp;"
    html += "<span style='display: inline-block; width: 12px; height: 12px; background: #fff9c4; border: 1px solid #ddd; margin-right: 4px;'></span> Different value</p>"

    return html


def get_consistency_matrix_plot(selected_indices):
    if not selected_indices or len(selected_indices) < 2:
        return None

    selected_snapshots = [state.snapshots[i] for i in selected_indices if i < len(state.snapshots)]
    if len(selected_snapshots) < 2:
        return None

    all_pairs = set()
    for s in selected_snapshots:
        for r in s["results"]:
            all_pairs.add(f"{r['cause']} -> {r['effect']}")
    all_pairs = sorted(list(all_pairs))

    if not all_pairs:
        return None

    n_pairs = len(all_pairs)
    n_snaps = len(selected_snapshots)

    pair_to_idx = {p: i for i, p in enumerate(all_pairs)}

    matrix = np.zeros((n_snaps, n_pairs))
    for si, s in enumerate(selected_snapshots):
        result_map = {}
        for r in s["results"]:
            key = f"{r['cause']} -> {r['effect']}"
            result_map[key] = r["is_significant"]
        for pi, pair in enumerate(all_pairs):
            sig = result_map.get(pair, None)
            if sig is True:
                matrix[si, pi] = 1
            elif sig is False:
                matrix[si, pi] = 0
            else:
                matrix[si, pi] = -1

    fig, ax = plt.subplots(figsize=(max(8, n_pairs * 0.6), max(4, n_snaps * 0.8)))

    cmap = plt.cm.colors.ListedColormap(['#cccccc', '#ef5350', '#66bb6a'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')

    ax.set_xticks(range(n_pairs))
    ax.set_xticklabels(all_pairs, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_snaps))
    ax.set_yticklabels([s['label'] for s in selected_snapshots], fontsize=10)
    ax.set_title("Causal Relationship Consistency Matrix", fontsize=12, fontweight='bold', pad=15)

    for i in range(n_snaps):
        for j in range(n_pairs):
            val = matrix[i, j]
            if val == 1:
                text = "✓"
                color = "white"
            elif val == 0:
                text = "✗"
                color = "white"
            else:
                text = "?"
                color = "#666"
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=10, fontweight='bold')

    legend_patches = [
        mpatches.Patch(color='#66bb6a', label='Significant'),
        mpatches.Patch(color='#ef5350', label='Not significant'),
        mpatches.Patch(color='#cccccc', label='Not tested')
    ]
    ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)

    plt.tight_layout()
    return fig


def get_statistic_trend_plot(selected_values, pair_selection):
    indices = _indices_from_selected(selected_values)
    if not indices or len(indices) < 2 or not pair_selection:
        return None

    selected_snapshots = [state.snapshots[i] for i in indices if i < len(state.snapshots)]
    if len(selected_snapshots) < 2:
        return None

    parts = pair_selection.split("->")
    if len(parts) != 2:
        return None
    cause = parts[0].strip()
    effect = parts[1].strip()

    snap_labels = []
    statistics = []
    p_values = []
    has_data = False

    for s in selected_snapshots:
        snap_labels.append(s["label"])
        found = False
        for r in s["results"]:
            if r["cause"] == cause and r["effect"] == effect:
                statistics.append(r["statistic"] if r["statistic"] is not None else 0)
                p_values.append(r["p_value"] if r["p_value"] is not None else None)
                found = True
                if r["statistic"] is not None:
                    has_data = True
                break
        if not found:
            statistics.append(0)
            p_values.append(None)

    if not has_data:
        return None

    fig, ax = plt.subplots(figsize=(max(6, len(snap_labels) * 1.2), 6))

    colors = []
    for pv in p_values:
        if pv is not None and pv < 0.05:
            colors.append('#66bb6a')
        elif pv is not None:
            colors.append('#ef5350')
        else:
            colors.append('#cccccc')

    x_pos = range(len(snap_labels))
    bars = ax.bar(x_pos, statistics, color=colors, edgecolor='#333', linewidth=0.5)

    for i, (bar, pv) in enumerate(zip(bars, p_values)):
        height = bar.get_height()
        if pv is not None:
            label = f"p={pv:.4f}"
        else:
            label = "N/A"
        ax.text(bar.get_x() + bar.get_width() / 2., height + (max(statistics) * 0.02 if max(statistics) > 0 else 0.1),
                label, ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(snap_labels, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel("Test Statistic", fontsize=11)
    ax.set_title(f"Statistic Trend: {cause} → {effect}", fontsize=12, fontweight='bold', pad=15)
    ax.set_ylim(0, max(statistics) * 1.2 if max(statistics) > 0 else 1)

    legend_patches = [
        mpatches.Patch(color='#66bb6a', label='Significant (p<0.05)'),
        mpatches.Patch(color='#ef5350', label='Not significant'),
        mpatches.Patch(color='#cccccc', label='No data')
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig


def on_compare_selected(selected_values):
    indices = _indices_from_selected(selected_values)
    if not indices or len(indices) < 2:
        return ("<p style='color: #888;'>Please select 2-4 snapshots to compare</p>",
                None, None, gr.update(choices=[], value=None))

    selected_count = len(indices)
    if selected_count > 4:
        return ("<p style='color: #e74c3c;'>Please select at most 4 snapshots for comparison</p>",
                None, None, gr.update(choices=[], value=None))

    param_html = get_param_diff_html(indices)
    consistency_fig = get_consistency_matrix_plot(indices)

    pair_options = _get_all_pair_options()
    first_pair = pair_options[0] if pair_options else None

    trend_fig = None
    if first_pair:
        trend_fig = get_statistic_trend_plot(indices, first_pair)

    return param_html, consistency_fig, trend_fig, gr.update(choices=pair_options, value=first_pair)


def build_app():
    with gr.Blocks(
        title="Causal Time Series Analysis",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css="""
        .gradio-container { max-width: 1400px !important; }
        """
    ) as app:

        gr.Markdown(
            """
            # Multivariate Time Series Causal Analysis
            Granger Causality | Transfer Entropy | PCMCI Causal Discovery
            """
        )

        with gr.Tabs():

            with gr.Tab("Data Import"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.File(label="Upload CSV File", file_types=[".csv"])
                        col_selector = gr.CheckboxGroup(
                            label="Select Variables for Analysis (2-10)",
                            choices=[], value=[]
                        )
                        data_info = gr.Textbox(label="Data Information", lines=12, interactive=False)
                    with gr.Column(scale=2):
                        preview_plot = gr.Plot(label="Time Series Preview")

                cond_cause = gr.Dropdown(label="Cause Variable (for Conditional)", choices=[], visible=False)
                cond_effect = gr.Dropdown(label="Effect Variable (for Conditional)", choices=[], visible=False)

                file_input.change(
                    fn=on_file_upload,
                    inputs=[file_input],
                    outputs=[col_selector, data_info, preview_plot]
                )
                col_selector.change(
                    fn=on_column_select,
                    inputs=[col_selector],
                    outputs=[data_info, preview_plot, cond_cause, cond_effect]
                )

            with gr.Tab("Preprocessing"):
                with gr.Row():
                    with gr.Column(scale=1):
                        missing_strategy = gr.Dropdown(
                            label="Missing Value Strategy",
                            choices=[
                                ("Linear Interpolation", "linear_interpolation"),
                                ("Forward Fill", "forward_fill"),
                                ("Drop Rows", "drop_rows")
                            ],
                            value="linear_interpolation"
                        )
                        standardize = gr.Dropdown(
                            label="Standardization",
                            choices=[
                                ("None", "none"),
                                ("Z-Score", "zscore"),
                                ("Min-Max", "minmax")
                            ],
                            value="none"
                        )
                        diff_order = gr.Dropdown(
                            label="Differencing Order",
                            choices=[("No Differencing", 0), ("1st Order", 1), ("2nd Order", 2)],
                            value=1
                        )
                        skip_stat = gr.Checkbox(
                            label="Skip Stationarity Check (use original series)",
                            value=False
                        )
                        preprocess_btn = gr.Button("Run Preprocessing", variant="primary")
                    with gr.Column(scale=1):
                        preprocess_info = gr.Textbox(label="Preprocessing Results", lines=14, interactive=False)
                        adf_plot = gr.Plot(label="ADF Test Visualization")

                preprocess_btn.click(
                    fn=on_preprocess,
                    inputs=[missing_strategy, standardize, diff_order, skip_stat],
                    outputs=[preprocess_info, adf_plot]
                )

            with gr.Tab("Granger Causality"):
                with gr.Tabs():
                    with gr.Tab("Bivariate"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                max_lag_bio = gr.Slider(1, 20, value=5, step=1, label="Max Lag Order")
                                criterion_bio = gr.Dropdown(
                                    label="Information Criterion",
                                    choices=[("AIC", "aic"), ("BIC", "bic"), ("HQIC", "hqic")],
                                    value="aic"
                                )
                                manual_lag = gr.Number(label="Manual Lag (0=auto)", value=0)
                                bio_btn = gr.Button("Run Bivariate Granger Test", variant="primary")
                                gr.Markdown("---")
                                gr.Markdown("### Save Snapshot")
                                bio_snap_label = gr.Textbox(label="Snapshot Label", placeholder="e.g., lag5_aic")
                                bio_snap_btn = gr.Button("Save Snapshot", variant="secondary")
                                bio_snap_status = gr.Textbox(label="Status", interactive=False, lines=2)
                            with gr.Column(scale=2):
                                bio_info = gr.Textbox(label="Results", lines=12, interactive=False)
                        with gr.Row():
                            bio_strength = gr.Plot(label="Causal Strength Matrix")
                            bio_heatmap = gr.Plot(label="Lag Heatmap")
                        with gr.Row():
                            pair_selector = gr.Dropdown(label="Select Pair for Scatter Plot", choices=[])
                            scatter_lag = gr.Slider(1, 20, value=10, step=1, label="Max Lag for Scatter")
                            scatter_btn = gr.Button("Plot Scatter")
                        scatter_plot = gr.Plot(label="Time Lag Scatter Plot")

                        bio_btn.click(
                            fn=on_bivariate_granger,
                            inputs=[max_lag_bio, criterion_bio, manual_lag],
                            outputs=[bio_info, bio_strength, bio_heatmap, pair_selector]
                        )
                        scatter_btn.click(
                            fn=on_lag_scatter,
                            inputs=[pair_selector, scatter_lag],
                            outputs=[scatter_plot]
                        )

                    with gr.Tab("Multivariate"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                max_lag_multi = gr.Slider(1, 20, value=5, step=1, label="Max Lag Order")
                                criterion_multi = gr.Dropdown(
                                    label="Information Criterion",
                                    choices=[("AIC", "aic"), ("BIC", "bic"), ("HQIC", "hqic")],
                                    value="aic"
                                )
                                multi_btn = gr.Button("Run Multivariate Granger Test", variant="primary")
                                gr.Markdown("---")
                                gr.Markdown("### Save Snapshot")
                                multi_snap_label = gr.Textbox(label="Snapshot Label", placeholder="e.g., multi_lag5")
                                multi_snap_btn = gr.Button("Save Snapshot", variant="secondary")
                                multi_snap_status = gr.Textbox(label="Status", interactive=False, lines=2)
                            with gr.Column(scale=2):
                                multi_info = gr.Textbox(label="Results", lines=14, interactive=False)
                        multi_plot = gr.Plot(label="Multivariate Wald Test Heatmap")

                        multi_btn.click(
                            fn=on_multivariate_granger,
                            inputs=[max_lag_multi, criterion_multi],
                            outputs=[multi_info, multi_plot]
                        )

                    with gr.Tab("Conditional"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                cond_cause_vis = gr.Dropdown(label="Cause Variable", choices=[])
                                cond_effect_vis = gr.Dropdown(label="Effect Variable", choices=[])
                                cond_control = gr.Textbox(
                                    label="Control Variables (comma-separated)",
                                    placeholder="e.g., var1, var2"
                                )
                                max_lag_cond = gr.Slider(1, 20, value=5, step=1, label="Max Lag Order")
                                criterion_cond = gr.Dropdown(
                                    label="Information Criterion",
                                    choices=[("AIC", "aic"), ("BIC", "bic"), ("HQIC", "hqic")],
                                    value="aic"
                                )
                                cond_btn = gr.Button("Run Conditional Granger Test", variant="primary")
                            with gr.Column(scale=2):
                                cond_info = gr.Textbox(label="Results", lines=10, interactive=False)

                        cond_btn.click(
                            fn=on_conditional_granger,
                            inputs=[cond_cause_vis, cond_effect_vis, cond_control, max_lag_cond, criterion_cond],
                            outputs=[cond_info]
                        )

                        col_selector.change(
                            fn=lambda cols: (
                                gr.update(choices=list(cols)),
                                gr.update(choices=list(cols))
                            ),
                            inputs=[col_selector],
                            outputs=[cond_cause_vis, cond_effect_vis]
                        )

            with gr.Tab("Transfer Entropy"):
                with gr.Row():
                    with gr.Column(scale=1):
                        te_embed_dim = gr.Slider(1, 5, value=1, step=1, label="Embedding Dimension")
                        te_k = gr.Slider(2, 20, value=4, step=1, label="K-Nearest Neighbors")
                        te_surrogates = gr.Slider(50, 500, value=100, step=50, label="Number of Surrogates")
                        te_btn = gr.Button("Run Transfer Entropy Analysis", variant="primary")
                        gr.Markdown("---")
                        gr.Markdown("### Save Snapshot")
                        te_snap_label = gr.Textbox(label="Snapshot Label", placeholder="e.g., te_dim2_k4")
                        te_snap_btn = gr.Button("Save Snapshot", variant="secondary")
                        te_snap_status = gr.Textbox(label="Status", interactive=False, lines=2)
                    with gr.Column(scale=2):
                        te_info = gr.Textbox(label="Results", lines=14, interactive=False)
                te_plot = gr.Plot(label="Transfer Entropy Heatmap")

                te_btn.click(
                    fn=on_transfer_entropy,
                    inputs=[te_embed_dim, te_k, te_surrogates],
                    outputs=[te_info, te_plot]
                )

            with gr.Tab("PCMCI Causal Graph"):
                with gr.Row():
                    with gr.Column(scale=1):
                        pcmci_tau = gr.Slider(1, 20, value=5, step=1, label="Max Time Lag (tau_max)")
                        pcmci_alpha = gr.Slider(0.01, 0.2, value=0.05, step=0.01, label="Significance Level (alpha)")
                        pcmci_ci = gr.Dropdown(
                            label="Conditional Independence Test",
                            choices=[("Partial Correlation (Linear)", "parcorr"), ("Mutual Information (Nonlinear)", "mi")],
                            value="parcorr"
                        )
                        pcmci_btn = gr.Button("Run PCMCI Algorithm", variant="primary")
                        gr.Markdown("---")
                        gr.Markdown("### Save Snapshot")
                        pcmci_snap_label = gr.Textbox(label="Snapshot Label", placeholder="e.g., pcmci_tau5_alpha05")
                        pcmci_snap_btn = gr.Button("Save Snapshot", variant="secondary")
                        pcmci_snap_status = gr.Textbox(label="Status", interactive=False, lines=2)
                    with gr.Column(scale=2):
                        pcmci_info = gr.Textbox(label="Results", lines=14, interactive=False)
                pcmci_graph = gr.Plot(label="PCMCI Causal Directed Graph")
                pcmci_heatmap = gr.Plot(label="PCMCI Strength Heatmap")

                pcmci_btn.click(
                    fn=on_pcmci,
                    inputs=[pcmci_tau, pcmci_alpha, pcmci_ci],
                    outputs=[pcmci_info, pcmci_graph, pcmci_heatmap]
                )

            with gr.Tab("Diagnostics"):
                with gr.Tabs():
                    with gr.Tab("Model Diagnostics"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                diag_lag = gr.Slider(1, 20, value=5, step=1, label="Max Lag Order")
                                diag_criterion = gr.Dropdown(
                                    label="Information Criterion",
                                    choices=[("AIC", "aic"), ("BIC", "bic"), ("HQIC", "hqic")],
                                    value="aic"
                                )
                                diag_btn = gr.Button("Run Diagnostics", variant="primary")
                            with gr.Column(scale=2):
                                diag_info = gr.Textbox(label="Results", lines=14, interactive=False)
                        diag_roots_plot = gr.Plot(label="VAR Eigenvalue Roots")

                        diag_btn.click(
                            fn=on_diagnostics,
                            inputs=[diag_lag, diag_criterion],
                            outputs=[diag_info, diag_roots_plot]
                        )

                    with gr.Tab("Multiple Comparison Correction"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                corr_method = gr.Dropdown(
                                    label="Correction Method",
                                    choices=[("Bonferroni", "bonferroni"), ("FDR (Benjamini-Hochberg)", "fdr")],
                                    value="bonferroni"
                                )
                                corr_alpha = gr.Slider(0.01, 0.1, value=0.05, step=0.01, label="Significance Level")
                                corr_btn = gr.Button("Apply Correction", variant="primary")
                            with gr.Column(scale=2):
                                corr_info = gr.Textbox(label="Corrected Results", lines=14, interactive=False)

                        corr_btn.click(
                            fn=on_correction,
                            inputs=[corr_method, corr_alpha],
                            outputs=[corr_info]
                        )

            with gr.Tab("Multi-Scale Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        ms_scales = gr.Textbox(
                            label="Time Scales (name:freq, comma-separated)",
                            value="Original:original, Weekly:W, Monthly:ME, Quarterly:QE",
                            lines=2
                        )
                        ms_lag = gr.Slider(1, 20, value=5, step=1, label="Max Lag Order")
                        ms_criterion = gr.Dropdown(
                            label="Information Criterion",
                            choices=[("AIC", "aic"), ("BIC", "bic"), ("HQIC", "hqic")],
                            value="aic"
                        )
                        ms_btn = gr.Button("Run Multi-Scale Analysis", variant="primary")
                    with gr.Column(scale=2):
                        ms_info = gr.Textbox(label="Results", lines=10, interactive=False)
                ms_plot = gr.Plot(label="Multi-Scale Comparison")

                ms_btn.click(
                    fn=on_multiscale,
                    inputs=[ms_scales, ms_lag, ms_criterion],
                    outputs=[ms_info, ms_plot]
                )

            with gr.Tab("Anomaly Root Cause"):
                with gr.Tabs():
                    with gr.Tab("Step 1: Anomaly Detection"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### Algorithm Parameters")
                                with gr.Accordion("Z-score Sliding Window", open=True):
                                    zscore_window = gr.Slider(10, 200, value=50, step=5, label="Window Size")
                                    zscore_threshold = gr.Slider(1.0, 5.0, value=3.0, step=0.5, label="Threshold (σ)")
                                with gr.Accordion("CUSUM Control Chart", open=True):
                                    cusum_k = gr.Slider(0.1, 2.0, value=0.5, step=0.1, label="Allowance k (×σ)")
                                    cusum_h = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="Decision Interval h (×σ)")
                                with gr.Accordion("Isolation Forest", open=True):
                                    if_contamination = gr.Slider(0.01, 0.2, value=0.05, step=0.01, label="Contamination Rate")
                                anomaly_btn = gr.Button("Run Anomaly Detection", variant="primary")
                                gr.Markdown("---")
                                gr.Markdown("### Detection Method for Root Cause")
                                anomaly_method = gr.Dropdown(
                                    label="Select Method",
                                    choices=[
                                        ("Consensus (≥2 algorithms)", "consensus"),
                                        ("Z-score (Sliding Window)", "zscore"),
                                        ("CUSUM", "cusum"),
                                        ("Isolation Forest", "iforest")
                                    ],
                                    value="consensus"
                                )
                                gr.Markdown("### Target Variable")
                                rc_target_var = gr.Dropdown(label="Target Variable (anomalous)", choices=[], value=None)
                            with gr.Column(scale=2):
                                anomaly_info = gr.Textbox(label="Detection Summary", lines=10, interactive=False)
                        with gr.Row():
                            anomaly_scatter = gr.Plot(label="Anomaly Scatter Plot")
                        with gr.Row():
                            anomaly_summary_table = gr.Dataframe(label="Summary Table", interactive=False)

                    with gr.Tab("Step 2: Root Cause Localization"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### Root Cause Parameters")
                                rc_window_size = gr.Slider(10, 100, value=30, step=5, label="Window Size (abnormal segment)")
                                rc_max_lag = gr.Slider(1, 20, value=5, step=1, label="Max Lag for Granger")
                                rc_criterion = gr.Dropdown(
                                    label="Information Criterion",
                                    choices=[("AIC", "aic"), ("BIC", "bic"), ("HQIC", "hqic")],
                                    value="aic"
                                )
                                rc_btn = gr.Button("Run Root Cause Analysis", variant="primary")
                            with gr.Column(scale=2):
                                rc_info = gr.Textbox(label="Root Cause Ranking", lines=12, interactive=False)
                        with gr.Row():
                            rc_bar = gr.Plot(label="Composite Score Bar Chart")
                        with gr.Row():
                            rc_timeline = gr.Plot(label="Anomaly Propagation Timeline")

                    with gr.Tab("Step 3: Visualization & Verification"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### Interactive Verification")
                                rc_candidate_select = gr.Dropdown(
                                    label="Select Candidate Root Cause Variable",
                                    choices=[], value=None
                                )
                                verify_btn = gr.Button("Show Verification Plots", variant="primary")
                        with gr.Row():
                            verify_scatter = gr.Plot(label="Scatter Comparison (Normal vs Abnormal)")
                        with gr.Row():
                            verify_ccf = gr.Plot(label="Cross-correlation Comparison")

                    with gr.Tab("Step 4: Propagation Path"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### PCMCI Causal Discovery Parameters")
                                pp_tau_max = gr.Slider(1, 20, value=5, step=1, label="Max Time Lag (tau_max)")
                                pp_alpha = gr.Slider(0.01, 0.2, value=0.05, step=0.01, label="Significance Level (alpha)")
                                pp_ci_test = gr.Dropdown(
                                    label="Conditional Independence Test",
                                    choices=[("Partial Correlation (Linear)", "parcorr"),
                                             ("Mutual Information (Nonlinear)", "mi")],
                                    value="parcorr"
                                )
                                gr.Markdown("---")
                                pp_btn = gr.Button("Infer Path", variant="primary", size="lg")
                                gr.Markdown(
                                    """
                                    💡 **How it works:**
                                    1. Runs PCMCI to discover the causal graph
                                    2. Filters edges where anomaly time-order matches (source anomalies precede target)
                                    3. BFS from root cause → target, sorted by avg causal strength
                                    4. Highlights the strongest path with thick dashed lines
                                    """
                                )
                            with gr.Column(scale=2):
                                pp_info = gr.Textbox(label="Path Analysis Results", lines=16, interactive=False)
                        with gr.Row():
                            pp_graph = gr.Plot(label="Anomaly Propagation Graph")
                        with gr.Row():
                            pp_summary = gr.Markdown(
                                value="<p style='color: #888; font-style: italic;'>Run 'Infer Path' to see the propagation path summary</p>"
                            )
                        with gr.Row():
                            pp_paths_list = gr.Dataframe(
                                label="All Discovered Paths (Top 3 by Avg Causal Strength)",
                                interactive=False,
                                headers=["Rank", "Path", "Total Lag", "Avg Causal Strength"],
                                datatype=["number", "str", "number", "number"]
                            )

                anomaly_btn.click(
                    fn=on_run_anomaly_detection,
                    inputs=[zscore_window, zscore_threshold, cusum_k, cusum_h, if_contamination],
                    outputs=[anomaly_info, anomaly_scatter, anomaly_summary_table, rc_target_var, rc_candidate_select]
                )
                anomaly_method.change(
                    fn=on_change_anomaly_method,
                    inputs=[anomaly_method],
                    outputs=[anomaly_scatter]
                )
                rc_btn.click(
                    fn=on_run_root_cause,
                    inputs=[rc_target_var, rc_window_size, rc_max_lag, rc_criterion],
                    outputs=[rc_info, rc_bar, rc_timeline, rc_candidate_select, verify_scatter]
                )
                rc_candidate_select.change(
                    fn=on_select_candidate_verification,
                    inputs=[rc_candidate_select],
                    outputs=[verify_scatter, verify_ccf]
                )
                verify_btn.click(
                    fn=on_select_candidate_verification,
                    inputs=[rc_candidate_select],
                    outputs=[verify_scatter, verify_ccf]
                )
                pp_btn.click(
                    fn=on_infer_propagation_path,
                    inputs=[pp_tau_max, pp_alpha, pp_ci_test],
                    outputs=[pp_info, pp_graph, pp_summary, pp_paths_list]
                )

            with gr.Tab("Snapshot Compare"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Saved Snapshots")
                        gr.Markdown("Select 2-4 snapshots to compare")
                        snap_list = gr.CheckboxGroup(
                            label="Snapshots",
                            choices=[],
                            value=[]
                        )
                        with gr.Row():
                            snap_compare_btn = gr.Button("Compare Selected", variant="primary", scale=2)
                            snap_delete_btn = gr.Button("Delete", variant="stop", scale=1)
                        snap_clear_btn = gr.Button("Clear All Snapshots", variant="secondary")
                        snap_status = gr.Textbox(label="Status", interactive=False, lines=3)
                    with gr.Column(scale=3):
                        with gr.Tabs():
                            with gr.Tab("Parameter Differences"):
                                param_diff_html = gr.HTML(
                                    value="<p style='color: #888;'>Select 2-4 snapshots to compare parameters</p>"
                                )
                            with gr.Tab("Consistency Matrix"):
                                consistency_plot = gr.Plot(label="Causal Relationship Consistency")
                            with gr.Tab("Statistic Trend"):
                                trend_pair_selector = gr.Dropdown(
                                    label="Select Variable Pair",
                                    choices=[],
                                    value=None
                                )
                                trend_plot = gr.Plot(label="Test Statistic Trend")

                snap_compare_btn.click(
                    fn=on_compare_selected,
                    inputs=[snap_list],
                    outputs=[param_diff_html, consistency_plot, trend_plot, trend_pair_selector]
                )
                snap_delete_btn.click(
                    fn=delete_snapshot,
                    inputs=[snap_list],
                    outputs=[snap_status, snap_list, trend_pair_selector, param_diff_html, consistency_plot, trend_plot]
                )
                snap_clear_btn.click(
                    fn=clear_all_snapshots,
                    outputs=[snap_status, snap_list, trend_pair_selector, param_diff_html, consistency_plot, trend_plot]
                )
                trend_pair_selector.change(
                    fn=get_statistic_trend_plot,
                    inputs=[snap_list, trend_pair_selector],
                    outputs=[trend_plot]
                )

            with gr.Tab("Report Export"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(
                            """
                            ### Generate PDF Analysis Report
                            The report will include:
                            - Data overview and statistics
                            - Stationarity test results
                            - Granger causality test results
                            - Causal graph and strength matrix
                            - Diagnostic information
                            - Conclusion summary
                            """
                        )
                        report_btn = gr.Button("Generate PDF Report", variant="primary", size="lg")
                        report_file = gr.File(label="Download Report")

                report_btn.click(
                    fn=on_generate_report,
                    outputs=[report_file]
                )

                bio_snap_btn.click(
                    fn=save_snapshot_bivariate,
                    inputs=[bio_snap_label, max_lag_bio, criterion_bio, manual_lag],
                    outputs=[bio_snap_status, snap_list, trend_pair_selector]
                )
                multi_snap_btn.click(
                    fn=save_snapshot_multivariate,
                    inputs=[multi_snap_label, max_lag_multi, criterion_multi],
                    outputs=[multi_snap_status, snap_list, trend_pair_selector]
                )
                te_snap_btn.click(
                    fn=save_snapshot_transfer_entropy,
                    inputs=[te_snap_label, te_embed_dim, te_k, te_surrogates],
                    outputs=[te_snap_status, snap_list, trend_pair_selector]
                )
                pcmci_snap_btn.click(
                    fn=save_snapshot_pcmci,
                    inputs=[pcmci_snap_label, pcmci_tau, pcmci_alpha, pcmci_ci],
                    outputs=[pcmci_snap_status, snap_list, trend_pair_selector]
                )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
