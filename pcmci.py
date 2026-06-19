import numpy as np
import pandas as pd
from itertools import combinations, product
from scipy import stats
from sklearn.covariance import EmpiricalCovariance


def partial_correlation(x, y, z_matrix, data):
    if z_matrix is None or len(z_matrix) == 0:
        r = np.corrcoef(data[x], data[y])[0, 1]
        return r

    cols = [x, y] + [c for c in z_matrix if c != x and c != y]
    sub = data[cols].dropna()
    if len(sub) < len(cols) + 5:
        return 0.0

    cov = np.cov(sub.T)
    try:
        prec = np.linalg.inv(cov)
        i = list(sub.columns).index(x)
        j = list(sub.columns).index(y)
        r = -prec[i, j] / np.sqrt(prec[i, i] * prec[j, j])
    except np.linalg.LinAlgError:
        r = 0.0
    return r


def _parcorr_test(x, y, z_matrix, data, alpha=0.05):
    r = partial_correlation(x, y, z_matrix, data)
    n = len(data[[x, y]].dropna())
    if n <= 3:
        return r, 1.0, False
    z_cols = list(z_matrix) if z_matrix is not None else []
    df = n - 2 - len(z_cols)
    if df <= 0:
        return r, 1.0, False
    t_stat = r * np.sqrt(df / (1 - r**2 + 1e-10))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    return r, p_value, p_value < alpha


def _mutual_info_estimate(x, y, z_matrix, data, n_neighbors=4):
    from scipy.spatial import KDTree
    from scipy.special import digamma

    cols = [x, y]
    if z_matrix is not None:
        cols += [c for c in z_matrix if c not in cols]
    sub = data[cols].dropna().values
    n = len(sub)
    if n <= n_neighbors + 1:
        return 0.0, 1.0, False

    x_idx, y_idx = 0, 1
    z_idx = list(range(2, len(cols))) if len(cols) > 2 else []

    x_data = sub[:, x_idx:x_idx+1]
    y_data = sub[:, y_idx:y_idx+1]
    z_data = sub[:, z_idx] if z_idx else None

    if z_data is not None:
        xz = np.hstack([x_data, z_data])
        yz = np.hstack([y_data, z_data])
        xyz = np.hstack([x_data, y_data, z_data])
    else:
        xz = x_data
        yz = y_data
        xyz = np.hstack([x_data, y_data])

    tree = KDTree(xyz)
    distances, _ = tree.query(xyz, k=n_neighbors+1)
    eps = np.maximum(distances[:, -1], 1e-10)

    tree_xz = KDTree(xz)
    nx = np.array([len(tree_xz.query_ball_point(xz[i], eps[i])) for i in range(n)])

    tree_yz = KDTree(yz)
    ny = np.array([len(tree_yz.query_ball_point(yz[i], eps[i])) for i in range(n)])

    mi = digamma(n) - np.mean(digamma(nx) + digamma(ny)) + digamma(n_neighbors)
    p_value = np.exp(-mi) if mi > 0 else 1.0
    p_value = min(1.0, max(0.0, p_value))
    return max(0.0, mi), p_value, p_value < 0.05


def pcmci_algorithm(df, selected_cols, tau_max=5, alpha=0.05, ci_test="parcorr"):
    data = df[selected_cols].dropna().reset_index(drop=True)
    n_vars = len(selected_cols)
    n_obs = len(data)

    parents = {i: [] for i in range(n_vars)}
    link_matrix = np.zeros((n_vars, n_vars, tau_max + 1))

    pc_parents = _pc_phase(data, selected_cols, tau_max, alpha, ci_test)

    mci_results = _mci_phase(data, selected_cols, pc_parents, tau_max, alpha, ci_test)

    edges = []
    for (j, i, tau), val in mci_results.items():
        if val["significant"]:
            link_matrix[i, j, tau] = val["strength"]
            edges.append({
                "source": selected_cols[j],
                "target": selected_cols[i],
                "lag": tau,
                "strength": round(val["strength"], 4),
                "p_value": round(val["p_value"], 4),
                "test_statistic": round(val["test_statistic"], 4)
            })

    return edges, link_matrix, pc_parents, mci_results


def _pc_phase(data, selected_cols, tau_max, alpha, ci_test):
    n_vars = len(selected_cols)
    parents = {i: set() for i in range(n_vars)}

    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue
            for tau in range(1, tau_max + 1):
                parents[i].add((j, tau))

    cond_set_size = 0
    max_cond_size = n_vars - 2

    while cond_set_size <= max_cond_size:
        removed = set()
        for i in range(n_vars):
            current_parents = parents[i].copy()
            for (j, tau) in current_parents:
                if (j, tau) in removed:
                    continue

                other_parents = [p for p in current_parents if p != (j, tau)]

                if len(other_parents) < cond_set_size:
                    continue

                for cond in combinations(other_parents, cond_set_size):
                    data_with_lag = data.copy()
                    z_cols = []
                    for (c, t) in cond:
                        col_name = f"{selected_cols[c]}_lag{t}"
                        data_with_lag[col_name] = data[selected_cols[c]].shift(t).values
                        z_cols.append(col_name)

                    x_name = selected_cols[i]
                    source_name = f"{selected_cols[j]}_lag{tau}"
                    data_with_lag[source_name] = data[selected_cols[j]].shift(tau).values

                    data_with_lag = data_with_lag.dropna()

                    if ci_test == "parcorr":
                        r, p, sig = _parcorr_test(
                            source_name, x_name, z_cols, data_with_lag, alpha
                        )
                    else:
                        mi, p, sig = _mutual_info_estimate(
                            source_name, x_name, z_cols, data_with_lag
                        )

                    if not sig:
                        parents[i].discard((j, tau))
                        removed.add((j, tau))
                        break

        cond_set_size += 1

    return parents


def _mci_phase(data, selected_cols, pc_parents, tau_max, alpha, ci_test):
    n_vars = len(selected_cols)
    mci_results = {}

    for i in range(n_vars):
        for (j, tau) in pc_parents[i]:
            data_test = data.copy()
            z_cols_target = []

            for (c, t) in pc_parents[i]:
                if (c, t) != (j, tau):
                    col_name = f"{selected_cols[c]}_lag{t}"
                    data_test[col_name] = data[selected_cols[c]].shift(t).values
                    z_cols_target.append(col_name)

            source_name = f"{selected_cols[j]}_lag{tau}"
            data_test[source_name] = data[selected_cols[j]].shift(tau).values

            data_test = data_test.dropna()

            if len(data_test) < 10:
                continue

            if ci_test == "parcorr":
                r, p, sig = _parcorr_test(
                    source_name, selected_cols[i],
                    z_cols_target, data_test, alpha
                )
                test_stat = r
                strength = abs(r)
            else:
                mi, p, sig = _mutual_info_estimate(
                    source_name, selected_cols[i],
                    z_cols_target, data_test
                )
                test_stat = mi
                strength = mi

            mci_results[(j, i, tau)] = {
                "strength": strength,
                "p_value": p,
                "test_statistic": test_stat,
                "significant": sig
            }

    return mci_results
