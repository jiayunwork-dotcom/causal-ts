import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox


def ljung_box_test(residuals_df, lags=10, significance=0.05):
    results = []
    for col in residuals_df.columns:
        series = residuals_df[col].dropna()
        if len(series) <= lags:
            results.append({
                "variable": col,
                "lb_statistic": np.nan,
                "lb_pvalue": np.nan,
                "is_white_noise": None,
                "warning": "Too few residuals for test"
            })
            continue
        lb_result = acorr_ljungbox(series, lags=lags, return_df=True)
        last_row = lb_result.iloc[-1]
        results.append({
            "variable": col,
            "lb_statistic": round(last_row["lb_stat"], 4),
            "lb_pvalue": round(last_row["lb_pvalue"], 4),
            "is_white_noise": last_row["lb_pvalue"] > significance,
            "warning": None
        })
    return pd.DataFrame(results)


def var_stability_check(fitted_var):
    try:
        roots = fitted_var.roots
        moduli = np.abs(roots)
        is_stable = all(m < 1.0 for m in moduli)
        return {
            "is_stable": is_stable,
            "max_modulus": round(float(np.max(moduli)), 4),
            "n_roots_outside": int(np.sum(moduli >= 1.0)),
            "root_moduli": [round(float(m), 4) for m in moduli]
        }
    except Exception as e:
        return {
            "is_stable": None,
            "max_modulus": np.nan,
            "n_roots_outside": np.nan,
            "root_moduli": [],
            "error": str(e)
        }


def bonferroni_correction(p_values, alpha=0.05):
    n_tests = len(p_values)
    if n_tests == 0:
        return [], []
    bonf_threshold = alpha / n_tests
    adjusted_p = [min(p * n_tests, 1.0) for p in p_values]
    significant = [p < alpha for p in adjusted_p]
    return adjusted_p, significant


def fdr_correction(p_values, alpha=0.05):
    n_tests = len(p_values)
    if n_tests == 0:
        return [], []

    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n_tests

    prev_adj = 0.0
    for rank_i, (orig_i, p_val) in enumerate(indexed_p):
        rank = rank_i + 1
        adj_p = min(p_val * n_tests / rank, 1.0)
        adj_p = max(adj_p, prev_adj)
        adjusted[orig_i] = adj_p
        prev_adj = adj_p

    significant = [p < alpha for p in adjusted]
    return adjusted, significant


def apply_multiple_comparison_correction(granger_df, method="bonferroni", alpha=0.05):
    if granger_df is None or len(granger_df) == 0:
        return granger_df

    df = granger_df.copy()
    p_col = "f_pvalue" if "f_pvalue" in df.columns else "wald_pvalue"

    p_values = df[p_col].tolist()
    p_values = [p if not np.isnan(p) else 1.0 for p in p_values]

    if method == "bonferroni":
        adjusted_p, significant = bonferroni_correction(p_values, alpha)
        method_name = "Bonferroni"
    else:
        adjusted_p, significant = fdr_correction(p_values, alpha)
        method_name = "FDR (Benjamini-Hochberg)"

    df[f"adjusted_pvalue_{method}"] = [round(p, 4) for p in adjusted_p]
    df[f"significant_{method}"] = significant

    return df


def run_full_diagnostics(df, selected_cols, max_lag, criterion="aic"):
    test_data = df[selected_cols].dropna()

    if len(test_data) < max_lag + 5:
        return None, None, None, "Insufficient data for diagnostics"

    try:
        optimal_lag, _ = select_optimal_lag_for_diag(test_data, max_lag, criterion)
        model = VAR(test_data)
        fitted = model.fit(optimal_lag)
    except Exception as e:
        return None, None, None, f"VAR model fitting failed: {str(e)}"

    residuals = fitted.resid
    lb_results = ljung_box_test(residuals)
    stability = var_stability_check(fitted)

    return fitted, lb_results, stability, None


def select_optimal_lag_for_diag(data, max_lag, criterion="aic"):
    best_lag = 1
    best_val = np.inf
    for lag in range(1, max_lag + 1):
        try:
            model = VAR(data)
            fitted = model.fit(lag)
            if criterion == "aic":
                val = fitted.aic
            elif criterion == "bic":
                val = fitted.bic
            elif criterion == "hqic":
                val = fitted.hqic
            else:
                val = fitted.aic
            if val < best_val:
                best_val = val
                best_lag = lag
        except Exception:
            continue
    return best_lag, None
