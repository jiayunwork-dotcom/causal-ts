import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def zscore_sliding_window(series, window_size=50, threshold=3.0):
    """
    基于滑动窗口的Z-score异常检测

    Parameters:
    -----------
    series : pd.Series
        单变量时间序列
    window_size : int
        滑动窗口大小
    threshold : float
        Z-score阈值倍数

    Returns:
    --------
    dict : 包含 anomaly_points (异常点索引列表), upper_bounds, lower_bounds
    """
    values = series.values
    n = len(values)

    upper_bounds = np.full(n, np.nan)
    lower_bounds = np.full(n, np.nan)
    is_anomaly = np.zeros(n, dtype=bool)

    for i in range(n):
        start = max(0, i - window_size + 1)
        window = values[start:i + 1]

        if len(window) < 2:
            continue

        mean = np.mean(window)
        std = np.std(window, ddof=1)

        if std == 0 or np.isnan(std):
            continue

        z = (values[i] - mean) / std
        upper_bounds[i] = mean + threshold * std
        lower_bounds[i] = mean - threshold * std

        if np.abs(z) > threshold:
            is_anomaly[i] = True

    anomaly_indices = np.where(is_anomaly)[0].tolist()

    return {
        "is_anomaly": is_anomaly,
        "anomaly_indices": anomaly_indices,
        "anomaly_count": len(anomaly_indices),
        "anomaly_ratio": len(anomaly_indices) / n if n > 0 else 0,
        "upper_bounds": upper_bounds,
        "lower_bounds": lower_bounds
    }


def cusum_detection(series, k=0.5, h=5.0):
    """
    基于CUSUM累积和控制图的异常检测

    Parameters:
    -----------
    series : pd.Series
        单变量时间序列
    k : float
        容许偏移量（参考值，通常为标准差的0.5-1倍）
    h : float
        决策区间（控制限）

    Returns:
    --------
    dict : 包含 anomaly_points, cusum_pos, cusum_neg
    """
    values = series.values
    n = len(values)

    if n < 2:
        return {
            "is_anomaly": np.zeros(n, dtype=bool),
            "anomaly_indices": [],
            "anomaly_count": 0,
            "anomaly_ratio": 0,
            "cusum_pos": np.zeros(n),
            "cusum_neg": np.zeros(n)
        }

    target = np.mean(values[:min(50, n)])
    sigma = np.std(values[:min(50, n)], ddof=1)
    if sigma == 0 or np.isnan(sigma):
        sigma = np.std(values, ddof=1)
    if sigma == 0 or np.isnan(sigma):
        sigma = 1.0

    k_abs = k * sigma
    h_abs = h * sigma

    cusum_pos = np.zeros(n)
    cusum_neg = np.zeros(n)
    is_anomaly = np.zeros(n, dtype=bool)

    for i in range(1, n):
        deviation = values[i] - target

        cusum_pos[i] = max(0, cusum_pos[i - 1] + deviation - k_abs)
        cusum_neg[i] = max(0, cusum_neg[i - 1] - deviation - k_abs)

        if cusum_pos[i] > h_abs or cusum_neg[i] > h_abs:
            is_anomaly[i] = True

    anomaly_indices = np.where(is_anomaly)[0].tolist()

    return {
        "is_anomaly": is_anomaly,
        "anomaly_indices": anomaly_indices,
        "anomaly_count": len(anomaly_indices),
        "anomaly_ratio": len(anomaly_indices) / n if n > 0 else 0,
        "cusum_pos": cusum_pos,
        "cusum_neg": cusum_neg,
        "h_abs": h_abs
    }


def isolation_forest_detection(series, contamination=0.05):
    """
    基于孤立森林的异常检测（单变量）

    Parameters:
    -----------
    series : pd.Series
        单变量时间序列
    contamination : float
        异常比例估计值

    Returns:
    --------
    dict : 包含 anomaly_points
    """
    values = series.values.reshape(-1, 1)
    n = len(values)

    if n < 10:
        return {
            "is_anomaly": np.zeros(n, dtype=bool),
            "anomaly_indices": [],
            "anomaly_count": 0,
            "anomaly_ratio": 0,
            "scores": np.zeros(n)
        }

    clf = IsolationForest(
        contamination=contamination,
        n_estimators=100,
        random_state=42
    )
    predictions = clf.fit_predict(values)
    scores = clf.decision_function(values)

    is_anomaly = predictions == -1
    anomaly_indices = np.where(is_anomaly)[0].tolist()

    return {
        "is_anomaly": is_anomaly,
        "anomaly_indices": anomaly_indices,
        "anomaly_count": len(anomaly_indices),
        "anomaly_ratio": len(anomaly_indices) / n if n > 0 else 0,
        "scores": scores
    }


def run_all_anomaly_detectors(df, selected_cols, zscore_window=50, zscore_threshold=3.0,
                              cusum_k=0.5, cusum_h=5.0, if_contamination=0.05):
    """
    对所有选中变量运行三种异常检测算法

    Returns:
    --------
    dict : {
        "per_variable": {col_name: {"zscore": ..., "cusum": ..., "iforest": ...}},
        "summary": DataFrame
    }
    """
    results = {}
    summary_rows = []

    for col in selected_cols:
        series = df[col].dropna()

        zscore_res = zscore_sliding_window(series, zscore_window, zscore_threshold)
        cusum_res = cusum_detection(series, cusum_k, cusum_h)
        iforest_res = isolation_forest_detection(series, if_contamination)

        results[col] = {
            "zscore": zscore_res,
            "cusum": cusum_res,
            "iforest": iforest_res,
            "series": series
        }

        summary_rows.append({"variable": col, "algorithm": "Z-score (Sliding Window)",
                             "anomaly_count": zscore_res["anomaly_count"],
                             "anomaly_ratio": round(zscore_res["anomaly_ratio"] * 100, 2)})
        summary_rows.append({"variable": col, "algorithm": "CUSUM",
                             "anomaly_count": cusum_res["anomaly_count"],
                             "anomaly_ratio": round(cusum_res["anomaly_ratio"] * 100, 2)})
        summary_rows.append({"variable": col, "algorithm": "Isolation Forest",
                             "anomaly_count": iforest_res["anomaly_count"],
                             "anomaly_ratio": round(iforest_res["anomaly_ratio"] * 100, 2)})

    summary_df = pd.DataFrame(summary_rows)

    return {
        "per_variable": results,
        "summary": summary_df
    }


def get_consensus_anomalies(anomaly_results, col, method="consensus"):
    """
    根据选择的方法获取最终的异常点

    Parameters:
    -----------
    anomaly_results : dict
        run_all_anomaly_detectors 的结果
    col : str
        变量名
    method : str
        "zscore", "cusum", "iforest", "consensus"

    Returns:
    --------
    np.array : 布尔数组，表示每个时间点是否为异常
    """
    var_res = anomaly_results["per_variable"][col]

    if method == "zscore":
        return var_res["zscore"]["is_anomaly"]
    elif method == "cusum":
        return var_res["cusum"]["is_anomaly"]
    elif method == "iforest":
        return var_res["iforest"]["is_anomaly"]
    elif method == "consensus":
        z_anom = var_res["zscore"]["is_anomaly"].astype(int)
        c_anom = var_res["cusum"]["is_anomaly"].astype(int)
        i_anom = var_res["iforest"]["is_anomaly"].astype(int)
        total = z_anom + c_anom + i_anom
        return total >= 2
    else:
        return var_res["zscore"]["is_anomaly"]


def get_anomaly_indices(anomaly_results, col, method="consensus"):
    """
    获取异常点的索引
    """
    is_anomaly = get_consensus_anomalies(anomaly_results, col, method)
    return np.where(is_anomaly)[0].tolist()
