import warnings
import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
from scipy import stats


warnings.filterwarnings("ignore")


def select_optimal_lag(data, max_lag, criterion="aic"):
    best_lag = 1
    best_val = np.inf
    results = {}
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
            results[lag] = val
            if val < best_val:
                best_val = val
                best_lag = lag
        except Exception:
            continue
    return best_lag, results


def bivariate_granger_test(df, selected_cols, max_lag, criterion="aic", manual_lag=None):
    pairs = list(combinations(selected_cols, 2))
    all_results = []

    for x_col, y_col in pairs:
        for cause, effect in [(x_col, y_col), (y_col, x_col)]:
            test_data = df[[effect, cause]].dropna()
            if len(test_data) < max_lag + 5:
                all_results.append({
                    "cause": cause, "effect": effect,
                    "optimal_lag": np.nan, "f_statistic": np.nan,
                    "f_pvalue": np.nan, "chi2_statistic": np.nan,
                    "chi2_pvalue": np.nan, "df_num": np.nan,
                    "df_denom": np.nan, "is_significant": None,
                    "error": "Insufficient data points"
                })
                continue

            if manual_lag is not None:
                lag = manual_lag
            else:
                lag, _ = select_optimal_lag(test_data, max_lag, criterion)

            try:
                gc_result = grangercausalitytests(
                    test_data, maxlag=lag, verbose=False
                )
                lag_result = gc_result[lag]
                ssr_ftest = lag_result[0]["ssr_ftest"]
                ssr_chi2test = lag_result[0]["ssr_chi2test"]

                all_results.append({
                    "cause": cause,
                    "effect": effect,
                    "optimal_lag": lag,
                    "f_statistic": round(ssr_ftest[0], 4),
                    "f_pvalue": round(ssr_ftest[1], 4),
                    "chi2_statistic": round(ssr_chi2test[0], 4),
                    "chi2_pvalue": round(ssr_chi2test[1], 4),
                    "df_num": ssr_ftest[2],
                    "df_denom": ssr_ftest[3],
                    "is_significant": ssr_ftest[1] < 0.05,
                    "error": None
                })
            except Exception as e:
                all_results.append({
                    "cause": cause, "effect": effect,
                    "optimal_lag": lag, "f_statistic": np.nan,
                    "f_pvalue": np.nan, "chi2_statistic": np.nan,
                    "chi2_pvalue": np.nan, "df_num": np.nan,
                    "df_denom": np.nan, "is_significant": None,
                    "error": str(e)
                })

    return pd.DataFrame(all_results)


def multivariate_granger_test(df, selected_cols, max_lag, criterion="aic"):
    test_data = df[selected_cols].dropna()
    if len(test_data) < max_lag + 5:
        return None, "Insufficient data for multivariate Granger test"

    optimal_lag, lag_results = select_optimal_lag(test_data, max_lag, criterion)

    try:
        model = VAR(test_data)
        fitted = model.fit(optimal_lag)
    except Exception as e:
        return None, f"VAR model fitting failed: {str(e)}"

    wald_results = []
    n_vars = len(selected_cols)
    p = optimal_lag
    k = n_vars

    try:
        params = fitted.params.values
        cov_params = fitted.cov_params()
        resid = fitted.resid.values
        n_obs = len(resid)
    except Exception as e:
        return None, f"Cannot extract model parameters: {str(e)}"

    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue
            cause_idx = j
            effect_idx = i

            try:
                param_indices = []
                for lag_idx in range(p):
                    row = 1 + lag_idx * k + cause_idx
                    col = effect_idx
                    flat_idx = row * k + col
                    if 0 <= flat_idx < params.size:
                        param_indices.append(flat_idx)

                if len(param_indices) == 0:
                    wald_stat = np.nan
                    wald_pval = np.nan
                else:
                    m = len(param_indices)
                    beta = params.flatten()[param_indices]
                    r = np.zeros((m, params.size))
                    for idx, pi in enumerate(param_indices):
                        r[idx, pi] = 1.0

                    r_cov = r @ cov_params @ r.T
                    try:
                        r_cov_inv = np.linalg.inv(r_cov)
                    except np.linalg.LinAlgError:
                        r_cov_inv = np.linalg.pinv(r_cov)

                    wald_stat = float(beta @ r_cov_inv @ beta)
                    df_num = m
                    df_denom = n_obs - (1 + p * k)
                    f_stat = wald_stat / df_num
                    wald_pval = float(1 - stats.f.cdf(f_stat, df_num, df_denom))

            except Exception:
                wald_stat = np.nan
                wald_pval = np.nan

            wald_results.append({
                "cause": selected_cols[cause_idx],
                "effect": selected_cols[effect_idx],
                "wald_statistic": round(wald_stat, 4) if not np.isnan(wald_stat) else np.nan,
                "wald_pvalue": round(wald_pval, 4) if not np.isnan(wald_pval) else np.nan,
                "is_significant": wald_pval < 0.05 if not np.isnan(wald_pval) else None,
                "optimal_lag": optimal_lag
            })

    return pd.DataFrame(wald_results), None


def conditional_granger_test(df, cause_col, effect_col, control_cols, max_lag, criterion="aic"):
    all_cols = [effect_col, cause_col] + list(control_cols)
    test_data = df[all_cols].dropna()

    if len(test_data) < max_lag + 5:
        return None, "Insufficient data for conditional Granger test"

    optimal_lag, _ = select_optimal_lag(test_data, max_lag, criterion)

    try:
        model = VAR(test_data)
        fitted = model.fit(optimal_lag)
    except Exception as e:
        return None, f"VAR model fitting failed: {str(e)}"

    n_vars = len(all_cols)
    p = optimal_lag
    k = n_vars
    effect_idx = 0
    cause_idx = 1

    try:
        params = fitted.params.values
        cov_params = fitted.cov_params()
        n_obs = len(fitted.resid)
    except Exception as e:
        return None, f"Cannot extract model parameters: {str(e)}"

    try:
        param_indices = []
        for lag_idx in range(p):
            row = 1 + lag_idx * k + cause_idx
            col = effect_idx
            flat_idx = row * k + col
            if 0 <= flat_idx < params.size:
                param_indices.append(flat_idx)

        if len(param_indices) == 0:
            wald_stat = np.nan
            wald_pval = np.nan
        else:
            m = len(param_indices)
            beta = params.flatten()[param_indices]
            r = np.zeros((m, params.size))
            for idx, pi in enumerate(param_indices):
                r[idx, pi] = 1.0

            r_cov = r @ cov_params @ r.T
            try:
                r_cov_inv = np.linalg.inv(r_cov)
            except np.linalg.LinAlgError:
                r_cov_inv = np.linalg.pinv(r_cov)

            wald_stat = float(beta @ r_cov_inv @ beta)
            df_num = m
            df_denom = n_obs - (1 + p * k)
            f_stat = wald_stat / df_num
            wald_pval = float(1 - stats.f.cdf(f_stat, df_num, df_denom))
    except Exception:
        wald_stat = np.nan
        wald_pval = np.nan

    result = {
        "cause": cause_col,
        "effect": effect_col,
        "controlling_for": ", ".join(control_cols),
        "wald_statistic": round(wald_stat, 4) if not np.isnan(wald_stat) else np.nan,
        "wald_pvalue": round(wald_pval, 4) if not np.isnan(wald_pval) else np.nan,
        "is_significant": wald_pval < 0.05 if not np.isnan(wald_pval) else None,
        "optimal_lag": optimal_lag
    }

    return result, None
