import numpy as np
import pandas as pd
from granger_test import bivariate_granger_test


def split_normal_abnormal_segments(df, target_col, anomaly_indices, window_size=30):
    """
    将时间序列按异常点切分为正常段和异常段

    Parameters:
    -----------
    df : pd.DataFrame
        完整数据
    target_col : str
        目标变量名
    anomaly_indices : list
        异常点索引列表
    window_size : int
        异常点前后窗口大小

    Returns:
    --------
    dict : {
        "normal_df": 正常段数据,
        "abnormal_df": 异常段数据,
        "normal_indices": 正常段索引,
        "abnormal_indices": 异常段索引
    }
    """
    n = len(df)

    anomaly_set = set(anomaly_indices)

    abnormal_window_indices = set()
    for ai in anomaly_indices:
        start = max(0, ai - window_size)
        end = min(n - 1, ai + window_size)
        for idx in range(start, end + 1):
            abnormal_window_indices.add(idx)

    normal_indices = sorted(set(range(n)) - abnormal_window_indices)
    abnormal_indices = sorted(abnormal_window_indices)

    normal_df = df.iloc[normal_indices].copy()
    abnormal_df = df.iloc[abnormal_indices].copy()

    return {
        "normal_df": normal_df,
        "abnormal_df": abnormal_df,
        "normal_indices": normal_indices,
        "abnormal_indices": abnormal_indices
    }


def compute_causal_strength_change(df, target_col, candidate_cols, anomaly_indices,
                                   window_size=30, max_lag=5, criterion="aic"):
    """
    计算正常段和异常段的Granger因果强度变化

    Parameters:
    -----------
    df : pd.DataFrame
        完整数据
    target_col : str
        目标变量（出现异常的变量）
    candidate_cols : list
        候选根因变量列表
    anomaly_indices : list
        目标变量的异常点索引
    window_size : int
        异常段窗口大小
    max_lag : int
        Granger检验最大滞后阶数
    criterion : str
        信息准则

    Returns:
    --------
    list : 每个候选变量的分析结果
    """
    segments = split_normal_abnormal_segments(df, target_col, anomaly_indices, window_size)
    normal_df = segments["normal_df"]
    abnormal_df = segments["abnormal_df"]

    results = []

    for candidate in candidate_cols:
        if candidate == target_col:
            continue

        normal_f = np.nan
        abnormal_f = np.nan
        change_rate = np.nan

        try:
            if len(normal_df) > max_lag + 5:
                normal_result = bivariate_granger_test(
                    normal_df, [candidate, target_col],
                    max_lag=max_lag, criterion=criterion
                )
                target_row = normal_result[
                    (normal_result["cause"] == candidate) &
                    (normal_result["effect"] == target_col)
                ]
                if len(target_row) > 0:
                    normal_f = target_row.iloc[0]["f_statistic"]
        except Exception:
            pass

        try:
            if len(abnormal_df) > max_lag + 5:
                abnormal_result = bivariate_granger_test(
                    abnormal_df, [candidate, target_col],
                    max_lag=max_lag, criterion=criterion
                )
                target_row = abnormal_result[
                    (abnormal_result["cause"] == candidate) &
                    (abnormal_result["effect"] == target_col)
                ]
                if len(target_row) > 0:
                    abnormal_f = target_row.iloc[0]["f_statistic"]
        except Exception:
            pass

        if not np.isnan(normal_f) and not np.isnan(abnormal_f) and normal_f > 0:
            change_rate = (abnormal_f - normal_f) / normal_f

        results.append({
            "candidate": candidate,
            "normal_f": round(normal_f, 4) if not np.isnan(normal_f) else np.nan,
            "abnormal_f": round(abnormal_f, 4) if not np.isnan(abnormal_f) else np.nan,
            "change_rate": round(change_rate, 4) if not np.isnan(change_rate) else np.nan
        })

    return results, segments


def compute_propagation_lag(target_anomaly_indices, candidate_anomaly_indices, max_lag=20):
    """
    计算异常传播时滞：候选根因变量的异常点平均比目标变量的异常点早多少个时间步

    Parameters:
    -----------
    target_anomaly_indices : list
        目标变量的异常点索引
    candidate_anomaly_indices : list
        候选根因变量的异常点索引
    max_lag : int
        最大考虑时滞

    Returns:
    --------
    float : 平均传播时滞（正数表示候选变量先异常）
    """
    if not target_anomaly_indices or not candidate_anomaly_indices:
        return np.nan

    target_set = set(target_anomaly_indices)
    candidate_set = set(candidate_anomaly_indices)

    lags = []
    for t_idx in target_anomaly_indices:
        best_lag = None
        for c_idx in candidate_anomaly_indices:
            lag = t_idx - c_idx
            if 0 <= lag <= max_lag:
                if best_lag is None or lag < best_lag:
                    best_lag = lag
        if best_lag is not None:
            lags.append(best_lag)

    if len(lags) == 0:
        return np.nan

    return float(np.mean(lags))


def root_cause_analysis(df, target_col, candidate_cols, anomaly_results,
                        anomaly_method="consensus", window_size=30,
                        max_lag=5, criterion="aic"):
    """
    完整的根因分析

    Parameters:
    -----------
    df : pd.DataFrame
        完整数据
    target_col : str
        目标变量
    candidate_cols : list
        候选根因变量
    anomaly_results : dict
        异常检测结果（来自 run_all_anomaly_detectors）
    anomaly_method : str
        使用的异常检测方法
    window_size : int
        异常段窗口大小
    max_lag : int
        Granger检验最大滞后
    criterion : str
        信息准则

    Returns:
    --------
    DataFrame : 根因排名表
    """
    from anomaly_detection import get_anomaly_indices

    target_anomalies = get_anomaly_indices(anomaly_results, target_col, anomaly_method)

    if not target_anomalies:
        return pd.DataFrame(), None

    causal_results, segments = compute_causal_strength_change(
        df, target_col, candidate_cols, target_anomalies,
        window_size=window_size, max_lag=max_lag, criterion=criterion
    )

    ranked_results = []
    for res in causal_results:
        candidate = res["candidate"]
        candidate_anomalies = get_anomaly_indices(anomaly_results, candidate, anomaly_method)

        prop_lag = compute_propagation_lag(target_anomalies, candidate_anomalies, max_lag=window_size)

        ranked_results.append({
            "candidate_variable": candidate,
            "normal_f_value": res["normal_f"],
            "abnormal_f_value": res["abnormal_f"],
            "change_rate": res["change_rate"],
            "propagation_lag": round(prop_lag, 2) if not np.isnan(prop_lag) else np.nan
        })

    ranked_df = pd.DataFrame(ranked_results)

    if len(ranked_df) > 0:
        valid_change = ranked_df[ranked_df["change_rate"].notna()]["change_rate"]
        if len(valid_change) > 0:
            cr_min, cr_max = valid_change.min(), valid_change.max()
            if cr_max != cr_min:
                ranked_df["norm_change_rate"] = (ranked_df["change_rate"] - cr_min) / (cr_max - cr_min)
            else:
                ranked_df["norm_change_rate"] = 0.5
        else:
            ranked_df["norm_change_rate"] = 0.5

        valid_lag = ranked_df[ranked_df["propagation_lag"].notna()]["propagation_lag"]
        if len(valid_lag) > 0:
            lag_min, lag_max = valid_lag.min(), valid_lag.max()
            if lag_max != lag_min:
                ranked_df["norm_lag_score"] = 1.0 - (ranked_df["propagation_lag"] - lag_min) / (lag_max - lag_min)
            else:
                ranked_df["norm_lag_score"] = 0.5
        else:
            ranked_df["norm_lag_score"] = 0.5

        ranked_df["norm_change_rate"] = ranked_df["norm_change_rate"].fillna(0)
        ranked_df["norm_lag_score"] = ranked_df["norm_lag_score"].fillna(0)

        ranked_df["composite_score"] = (
            ranked_df["norm_change_rate"] * 0.6 +
            ranked_df["norm_lag_score"] * 0.4
        ).round(4)

        ranked_df = ranked_df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    return ranked_df, segments
