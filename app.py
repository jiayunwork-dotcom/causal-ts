import os
import tempfile
import warnings
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
