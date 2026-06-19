import os
import io
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT


def _df_to_table(df, max_rows=None):
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)

    data = [df.columns.tolist()] + df.values.tolist()
    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("FONTSIZE", (0, 1), (-1, -1), 7),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4ff")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ])

    n_cols = len(df.columns)
    for col_idx in range(n_cols):
        style.add("LEFTPADDING", (col_idx, 0), (col_idx, -1), 4)
        style.add("RIGHTPADDING", (col_idx, 0), (col_idx, -1), 4)

    return Table(data, style=style)


def generate_pdf_report(
    stats_df,
    adf_df,
    granger_df,
    multivar_df,
    te_results,
    pcmci_edges,
    lb_df,
    stability_info,
    selected_cols,
    figures=None,
    output_path=None
):
    if output_path is None:
        output_path = os.path.join(tempfile.gettempdir(), f"causal_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        rightMargin=1.5*cm, leftMargin=1.5*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("CustomTitle", parent=styles["Title"],
                                  fontSize=18, textColor=colors.HexColor("#1e40af"),
                                  spaceAfter=6)
    heading_style = ParagraphStyle("CustomHeading", parent=styles["Heading2"],
                                    fontSize=13, textColor=colors.HexColor("#1e40af"),
                                    spaceAfter=6, spaceBefore=12)
    body_style = ParagraphStyle("CustomBody", parent=styles["Normal"],
                                 fontSize=9, spaceAfter=4)
    small_style = ParagraphStyle("SmallText", parent=styles["Normal"],
                                  fontSize=8, textColor=colors.grey)

    elements = []

    elements.append(Paragraph("Multivariate Time Series Causal Analysis Report", title_style))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", small_style))
    elements.append(Paragraph(f"Variables: {', '.join(selected_cols)}", body_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1e40af")))
    elements.append(Spacer(1, 12))

    if stats_df is not None and len(stats_df) > 0:
        elements.append(Paragraph("1. Data Overview", heading_style))
        elements.append(Paragraph(f"Number of variables: {len(selected_cols)}", body_style))
        elements.append(Spacer(1, 6))
        elements.append(_df_to_table(stats_df))
        elements.append(Spacer(1, 12))

    if figures and "preview" in figures:
        elements.append(Paragraph("Time Series Preview", heading_style))
        try:
            elements.append(Image(figures["preview"], width=16*cm, height=8*cm))
        except Exception:
            elements.append(Paragraph("[Preview figure not available]", body_style))
        elements.append(Spacer(1, 12))

    if adf_df is not None and len(adf_df) > 0:
        elements.append(Paragraph("2. Stationarity Test Results (ADF)", heading_style))
        elements.append(_df_to_table(adf_df))
        n_stationary = adf_df["is_stationary"].sum() if "is_stationary" in adf_df.columns else 0
        elements.append(Paragraph(
            f"Stationary: {n_stationary}/{len(adf_df)} variables",
            body_style
        ))
        elements.append(Spacer(1, 12))

    if granger_df is not None and len(granger_df) > 0:
        elements.append(PageBreak())
        elements.append(Paragraph("3. Bivariate Granger Causality Test", heading_style))
        sig_count = granger_df["is_significant"].sum() if "is_significant" in granger_df.columns else 0
        elements.append(Paragraph(
            f"Significant causal pairs: {sig_count}/{len(granger_df)}",
            body_style
        ))
        elements.append(Spacer(1, 6))
        elements.append(_df_to_table(granger_df, max_rows=50))
        elements.append(Spacer(1, 12))

    if figures and "causal_graph" in figures:
        elements.append(Paragraph("Causal Graph", heading_style))
        try:
            elements.append(Image(figures["causal_graph"], width=14*cm, height=10*cm))
        except Exception:
            elements.append(Paragraph("[Causal graph not available]", body_style))
        elements.append(Spacer(1, 12))

    if figures and "strength_matrix" in figures:
        elements.append(Paragraph("Causal Strength Matrix", heading_style))
        try:
            elements.append(Image(figures["strength_matrix"], width=14*cm, height=10*cm))
        except Exception:
            elements.append(Paragraph("[Strength matrix not available]", body_style))
        elements.append(Spacer(1, 12))

    if multivar_df is not None and len(multivar_df) > 0:
        elements.append(PageBreak())
        elements.append(Paragraph("4. Multivariate Granger Test (VAR Wald)", heading_style))
        elements.append(_df_to_table(multivar_df, max_rows=50))
        elements.append(Spacer(1, 12))

    if te_results and len(te_results) > 0:
        elements.append(Paragraph("5. Transfer Entropy Results", heading_style))
        te_df = pd.DataFrame(te_results)
        elements.append(_df_to_table(te_df, max_rows=50))
        elements.append(Spacer(1, 12))

    if pcmci_edges and len(pcmci_edges) > 0:
        elements.append(Paragraph("6. PCMCI Causal Discovery", heading_style))
        pcmci_df = pd.DataFrame(pcmci_edges)
        elements.append(_df_to_table(pcmci_df))
        elements.append(Spacer(1, 12))

    if lb_df is not None and len(lb_df) > 0:
        elements.append(PageBreak())
        elements.append(Paragraph("7. Diagnostics", heading_style))
        elements.append(Paragraph("Ljung-Box Residual Test", heading_style))
        elements.append(_df_to_table(lb_df))
        elements.append(Spacer(1, 8))

        if stability_info:
            elements.append(Paragraph("VAR Model Stability", heading_style))
            stable = stability_info.get("is_stable", None)
            status_text = "STABLE" if stable else "UNSTABLE"
            status_color = "#16a34a" if stable else "#dc2626"
            elements.append(Paragraph(
                f"Status: <font color='{status_color}'>{status_text}</font>",
                body_style
            ))
            elements.append(Paragraph(
                f"Max eigenvalue modulus: {stability_info.get('max_modulus', 'N/A')}",
                body_style
            ))
            elements.append(Spacer(1, 12))

    elements.append(PageBreak())
    elements.append(Paragraph("8. Conclusion Summary", heading_style))

    conclusions = []
    if granger_df is not None and len(granger_df) > 0:
        sig = granger_df[granger_df["is_significant"] == True] if "is_significant" in granger_df.columns else pd.DataFrame()
        if len(sig) > 0:
            for _, row in sig.iterrows():
                conclusions.append(
                    f"• {row['cause']} → {row['effect']} (F={row.get('f_statistic', 'N/A')}, "
                    f"p={row.get('f_pvalue', 'N/A')}, lag={row.get('optimal_lag', 'N/A')})"
                )
        else:
            conclusions.append("• No significant Granger causal relationships found at p < 0.05")

    if pcmci_edges and len(pcmci_edges) > 0:
        conclusions.append(f"• PCMCI identified {len(pcmci_edges)} causal edges")
    else:
        conclusions.append("• PCMCI did not identify significant causal edges")

    if te_results and len(te_results) > 0:
        sig_te = [r for r in te_results if r.get("is_significant", False)]
        conclusions.append(f"• Transfer entropy found {len(sig_te)} significant nonlinear causal links")

    for c in conclusions:
        elements.append(Paragraph(c, body_style))

    doc.build(elements)
    return output_path
