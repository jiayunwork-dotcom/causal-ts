import numpy as np
import pandas as pd
from granger_test import bivariate_granger_test


def _cluster_anomaly_events(anomaly_indices, gap_threshold=10):
    """
    将异常点聚类为异常事件

    Parameters:
    -----------
    anomaly_indices : list
        异常点索引列表（已排序）
    gap_threshold : int
        两个异常点之间的间隔超过此值则视为不同事件

    Returns:
    --------
    list : 每个元素是 (start_idx, end_idx) 表示一个异常事件的起止索引
    """
    if not anomaly_indices:
        return []

    sorted_indices = sorted(anomaly_indices)
    events = []
    current_start = sorted_indices[0]
    current_end = sorted_indices[0]

    for i in range(1, len(sorted_indices)):
        if sorted_indices[i] - current_end <= gap_threshold:
            current_end = sorted_indices[i]
        else:
            events.append((current_start, current_end))
            current_start = sorted_indices[i]
            current_end = sorted_indices[i]

    events.append((current_start, current_end))
    return events


def split_normal_abnormal_segments(df, target_col, anomaly_indices, window_size=30):
    """
    将时间序列按异常事件切分为正常段和异常段

    策略：
    - 将异常点聚类为异常事件
    - 正常段：每个异常事件之前的 window_size 个点（异常发生前的正常时期）
    - 异常段：每个异常事件及之后的 window_size 个点

    Parameters:
    -----------
    df : pd.DataFrame
        完整数据
    target_col : str
        目标变量名
    anomaly_indices : list
        异常点索引列表
    window_size : int
        窗口大小

    Returns:
    --------
    dict
    """
    n = len(df)

    if not anomaly_indices:
        return {
            "normal_df": df.copy(),
            "abnormal_df": df.iloc[0:0].copy(),
            "normal_indices": list(range(n)),
            "abnormal_indices": [],
            "events": []
        }

    events = _cluster_anomaly_events(anomaly_indices, gap_threshold=max(5, window_size // 3))

    normal_indices_set = set()
    abnormal_indices_set = set()

    for start_idx, end_idx in events:
        pre_start = max(0, start_idx - window_size)
        pre_end = max(0, start_idx - 1)
        if pre_start <= pre_end:
            for idx in range(pre_start, pre_end + 1):
                normal_indices_set.add(idx)

        post_start = start_idx
        post_end = min(n - 1, end_idx + window_size)
        if post_start <= post_end:
            for idx in range(post_start, post_end + 1):
                abnormal_indices_set.add(idx)

    min_normal_size = max(20, window_size)
    if len(normal_indices_set) < min_normal_size:
        all_normal = set(range(n)) - abnormal_indices_set
        if len(all_normal) > min_normal_size:
            normal_indices_set = all_normal

    normal_indices = sorted(normal_indices_set)
    abnormal_indices = sorted(abnormal_indices_set)

    normal_df = df.iloc[normal_indices].copy()
    abnormal_df = df.iloc[abnormal_indices].copy()

    return {
        "normal_df": normal_df,
        "abnormal_df": abnormal_df,
        "normal_indices": normal_indices,
        "abnormal_indices": abnormal_indices,
        "events": events
    }


def compute_causal_strength_change(df, target_col, candidate_cols, anomaly_indices,
                                   window_size=30, max_lag=5, criterion="aic"):
    """
    计算正常段和异常段的Granger因果强度变化
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
                    val = target_row.iloc[0]["f_statistic"]
                    if not pd.isna(val):
                        normal_f = float(val)
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
                    val = target_row.iloc[0]["f_statistic"]
                    if not pd.isna(val):
                        abnormal_f = float(val)
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
    """
    if not target_anomaly_indices or not candidate_anomaly_indices:
        return np.nan

    candidate_set = set(candidate_anomaly_indices)

    lags = []
    for t_idx in target_anomaly_indices:
        best_lag = None
        for lag in range(0, max_lag + 1):
            if (t_idx - lag) in candidate_set:
                best_lag = lag
                break
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
        has_valid_change = ranked_df["change_rate"].notna().any()

        if has_valid_change:
            valid_change = ranked_df[ranked_df["change_rate"].notna()]["change_rate"]
            cr_min, cr_max = valid_change.min(), valid_change.max()
            if cr_max != cr_min:
                ranked_df["norm_change_rate"] = (ranked_df["change_rate"] - cr_min) / (cr_max - cr_min)
                ranked_df["norm_change_rate"] = ranked_df["norm_change_rate"].fillna(0)
            else:
                ranked_df["norm_change_rate"] = 0.5
        else:
            valid_abnormal_f = ranked_df[ranked_df["abnormal_f_value"].notna()]["abnormal_f_value"]
            if len(valid_abnormal_f) > 0:
                af_min, af_max = valid_abnormal_f.min(), valid_abnormal_f.max()
                if af_max != af_min:
                    ranked_df["norm_change_rate"] = (ranked_df["abnormal_f_value"] - af_min) / (af_max - af_min)
                    ranked_df["norm_change_rate"] = ranked_df["norm_change_rate"].fillna(0)
                else:
                    ranked_df["norm_change_rate"] = 0.5
            else:
                ranked_df["norm_change_rate"] = 0.5

        valid_lag = ranked_df[ranked_df["propagation_lag"].notna()]["propagation_lag"]
        if len(valid_lag) > 0:
            lag_min, lag_max = valid_lag.min(), valid_lag.max()
            if lag_max != lag_min:
                ranked_df["norm_lag_score"] = 1.0 - (ranked_df["propagation_lag"] - lag_min) / (lag_max - lag_min)
                ranked_df["norm_lag_score"] = ranked_df["norm_lag_score"].fillna(0)
            else:
                ranked_df["norm_lag_score"] = 0.5
        else:
            ranked_df["norm_lag_score"] = 0.5

        ranked_df["composite_score"] = (
            ranked_df["norm_change_rate"] * 0.6 +
            ranked_df["norm_lag_score"] * 0.4
        ).round(4)

        ranked_df = ranked_df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    return ranked_df, segments
