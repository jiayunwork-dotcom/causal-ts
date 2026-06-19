import numpy as np
from scipy.spatial import KDTree
from scipy.special import digamma
from itertools import combinations


def _ksg_entropy(data, k=4):
    n = len(data)
    if n <= k:
        return 0.0
    tree = KDTree(data)
    distances, _ = tree.query(data, k=k+1)
    eps = distances[:, -1]
    eps = np.maximum(eps, 1e-10)
    return digamma(n) - digamma(k) + np.mean(np.log(eps))


def _ksg_conditional_mutual_info(x, y, z, k=4):
    n = len(x)
    if n <= k + 1:
        return 0.0

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1) if z is not None else None

    if z is not None:
        xz = np.hstack([x, z])
        yz = np.hstack([y, z])
        xyz = np.hstack([x, y, z])
    else:
        xz = x
        yz = y
        xyz = np.hstack([x, y])

    tree_xyz = KDTree(xyz)
    distances, _ = tree_xyz.query(xyz, k=k+1)
    eps = np.maximum(distances[:, -1], 1e-10)

    tree_xz = KDTree(xz)
    nx = np.array([len(tree_xz.query_ball_point(xz[i], eps[i])) for i in range(n)])

    tree_yz = KDTree(yz)
    ny = np.array([len(tree_yz.query_ball_point(yz[i], eps[i])) for i in range(n)])

    te = digamma(k) - np.mean(digamma(nx) + digamma(ny)) + digamma(n)
    return max(0.0, te)


def compute_transfer_entropy(source, target, embedding_dim=1, k=4, lag=1):
    n = len(source)
    max_start = n - lag - embedding_dim
    if max_start <= k + 1:
        return 0.0

    target_future = target[lag + embedding_dim:]
    target_past_list = []
    source_past_list = []

    for d in range(embedding_dim):
        target_past_list.append(target[lag + d: lag + d + max_start])
        source_past_list.append(source[d: d + max_start])

    target_future = target_future[:max_start]
    target_past = np.column_stack(target_past_list)
    source_past = np.column_stack(source_past_list)

    te = _ksg_conditional_mutual_info(source_past, target_future.reshape(-1, 1), target_past, k=k)
    return te


def surrogate_significance_test(source, target, embedding_dim=1, k=4, lag=1, n_surrogates=100):
    original_te = compute_transfer_entropy(source, target, embedding_dim, k, lag)

    n = len(source)
    surrogate_tes = []
    rng = np.random.RandomState(42)

    for _ in range(n_surrogates):
        shuffled_source = rng.permutation(source)
        te_s = compute_transfer_entropy(shuffled_source, target, embedding_dim, k, lag)
        surrogate_tes.append(te_s)

    surrogate_tes = np.array(surrogate_tes)
    p_value = np.mean(surrogate_tes >= original_te)

    return original_te, p_value, surrogate_tes


def pairwise_transfer_entropy(df, selected_cols, embedding_dim=1, k=4, lag=1, n_surrogates=100):
    pairs = list(combinations(selected_cols, 2))
    results = []

    for x_col, y_col in pairs:
        source = df[x_col].dropna().values
        target = df[y_col].dropna().values
        min_len = min(len(source), len(target))
        source = source[:min_len]
        target = target[:min_len]

        te_xy, p_xy, surr_xy = surrogate_significance_test(
            source, target, embedding_dim, k, lag, n_surrogates
        )
        te_yx, p_yx, surr_yx = surrogate_significance_test(
            target, source, embedding_dim, k, lag, n_surrogates
        )

        results.append({
            "source": x_col,
            "target": y_col,
            "transfer_entropy": round(te_xy, 6),
            "p_value": round(p_xy, 4),
            "is_significant": p_xy < 0.05,
            "surrogate_mean": round(np.mean(surr_xy), 6),
            "surrogate_std": round(np.std(surr_xy), 6)
        })
        results.append({
            "source": y_col,
            "target": x_col,
            "transfer_entropy": round(te_yx, 6),
            "p_value": round(p_yx, 4),
            "is_significant": p_yx < 0.05,
            "surrogate_mean": round(np.mean(surr_yx), 6),
            "surrogate_std": round(np.std(surr_yx), 6)
        })

    return results
