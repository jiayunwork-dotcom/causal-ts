import numpy as np
from collections import deque
from pcmci import pcmci_algorithm


def build_anomaly_propagation_graph(df, selected_cols, pcmci_edges,
                                     anomaly_results, anomaly_method="consensus"):
    """
    构建异常传播子图：从完整PCMCI因果图中筛选出异常有沿边传播迹象的边
    
    筛选规则：对于因果边 A->B，如果 A 的异常点平均比 B 的异常点早出现（时滞>0），则保留
    
    Parameters:
    -----------
    df : pd.DataFrame
        预处理后的数据
    selected_cols : list
        选中的变量列表
    pcmci_edges : list
        pcmci_algorithm 返回的边列表
    anomaly_results : dict
        run_all_anomaly_detectors 的结果
    anomaly_method : str
        异常检测方法
    
    Returns:
    --------
    list : 异常传播子图的边列表，每条边包含 source, target, lag, strength, p_value,
           anomaly_lag (A异常点平均领先B异常点的时滞)
    """
    from anomaly_detection import get_anomaly_indices
    
    var_anomaly_indices = {}
    for col in selected_cols:
        var_anomaly_indices[col] = get_anomaly_indices(anomaly_results, col, anomaly_method)
    
    def _compute_pair_anomaly_lag(source_indices, target_indices, max_gap=50):
        """计算 source 异常点平均比 target 异常点早多少步"""
        if not source_indices or not target_indices:
            return np.nan
        
        source_set = set(source_indices)
        lags = []
        for t_idx in target_indices:
            best_lag = None
            for lag in range(0, max_gap + 1):
                if (t_idx - lag) in source_set:
                    best_lag = lag
                    break
            if best_lag is not None:
                lags.append(best_lag)
        
        if len(lags) == 0:
            return np.nan
        return float(np.mean(lags))
    
    propagation_edges = []
    
    for edge in pcmci_edges:
        source = edge["source"]
        target = edge["target"]
        
        if source not in var_anomaly_indices or target not in var_anomaly_indices:
            continue
        
        anomaly_lag = _compute_pair_anomaly_lag(
            var_anomaly_indices[source],
            var_anomaly_indices[target]
        )
        
        if not np.isnan(anomaly_lag) and anomaly_lag > 0:
            propagation_edges.append({
                "source": source,
                "target": target,
                "lag": edge["lag"],
                "strength": edge["strength"],
                "p_value": edge.get("p_value", np.nan),
                "anomaly_lag": round(anomaly_lag, 2)
            })
    
    return propagation_edges


def find_all_paths_bfs(propagation_edges, root_cause, target_var, max_path_len=None):
    """
    使用广度优先搜索找到从 root_cause 到 target_var 的所有简单路径
    
    正确处理有环图：记录每条路径的已访问节点，防止死循环
    
    Parameters:
    -----------
    propagation_edges : list
        异常传播子图的边列表
    root_cause : str
        根因变量（起点）
    target_var : str
        目标变量（终点）
    max_path_len : int or None
        最大路径长度（边数），默认等于变量总数
    
    Returns:
    --------
    list : 所有路径列表，每条路径是一个 dict，包含：
           - nodes: 节点列表 [root, ..., target]
           - edges: 边属性列表 [{source, target, lag, strength, anomaly_lag}, ...]
           - avg_strength: 路径上各边的平均因果强度
           - total_lag: 路径上各边的 anomaly_lag 之和
    """
    adjacency = {}
    edge_info = {}
    
    for edge in propagation_edges:
        src = edge["source"]
        tgt = edge["target"]
        if src not in adjacency:
            adjacency[src] = []
        adjacency[src].append(tgt)
        edge_info[(src, tgt)] = edge
    
    all_vars_in_graph = set()
    for e in propagation_edges:
        all_vars_in_graph.add(e["source"])
        all_vars_in_graph.add(e["target"])
    
    if max_path_len is None:
        max_path_len = len(all_vars_in_graph)
    
    if root_cause == target_var:
        return []
    
    if root_cause not in adjacency:
        return []
    
    all_paths = []
    
    queue = deque()
    initial_path = [root_cause]
    queue.append((initial_path, set(initial_path)))
    
    while queue:
        current_path, visited = queue.popleft()
        current_node = current_path[-1]
        
        if current_node == target_var:
            path_edges = []
            strengths = []
            total_anomaly_lag = 0.0
            for i in range(len(current_path) - 1):
                s, t = current_path[i], current_path[i + 1]
                e_info = edge_info[(s, t)]
                path_edges.append(e_info)
                strengths.append(e_info["strength"])
                total_anomaly_lag += e_info["anomaly_lag"]
            
            avg_strength = float(np.mean(strengths)) if strengths else 0.0
            
            all_paths.append({
                "nodes": current_path,
                "edges": path_edges,
                "avg_strength": round(avg_strength, 4),
                "total_lag": round(total_anomaly_lag, 2)
            })
            continue
        
        if len(current_path) - 1 >= max_path_len:
            continue
        
        if current_node not in adjacency:
            continue
        
        for neighbor in adjacency[current_node]:
            if neighbor not in visited:
                new_path = current_path + [neighbor]
                new_visited = visited | {neighbor}
                queue.append((new_path, new_visited))
    
    all_paths.sort(key=lambda p: p["avg_strength"], reverse=True)
    
    return all_paths


def select_top_paths(all_paths, top_k=3):
    """
    选取平均因果强度最强的前 K 条路径
    
    Parameters:
    -----------
    all_paths : list
        find_all_paths_bfs 返回的路径列表
    top_k : int
        返回的路径数量
    
    Returns:
    --------
    list : 前 K 条路径
    """
    return all_paths[:top_k]


def run_propagation_path_analysis(df, selected_cols, root_cause, target_var,
                                   anomaly_results, pcmci_params,
                                   anomaly_method="consensus", top_k=3):
    """
    完整的异常传播路径分析流程
    
    Parameters:
    -----------
    df : pd.DataFrame
        预处理后的数据
    selected_cols : list
        选中的变量列表
    root_cause : str
        综合评分最高的根因变量
    target_var : str
        目标变量
    anomaly_results : dict
        异常检测结果
    pcmci_params : dict
        PCMCI 参数 {"tau_max": int, "alpha": float, "ci_test": str}
    anomaly_method : str
        异常检测方法
    top_k : int
        返回的最强路径数量
    
    Returns:
    --------
    dict : {
        "pcmci_edges": 完整的 PCMCI 因果边,
        "propagation_edges": 异常传播子图边,
        "all_paths": 所有找到的路径,
        "top_paths": 前 K 条最强路径,
        "strongest_path": 最强路径 (top_paths[0] if exists else None)
    }
    """
    pcmci_edges, link_matrix, pc_parents, mci_results = pcmci_algorithm(
        df, selected_cols,
        tau_max=pcmci_params.get("tau_max", 5),
        alpha=pcmci_params.get("alpha", 0.05),
        ci_test=pcmci_params.get("ci_test", "parcorr")
    )
    
    propagation_edges = build_anomaly_propagation_graph(
        df, selected_cols, pcmci_edges, anomaly_results, anomaly_method
    )
    
    all_paths = find_all_paths_bfs(
        propagation_edges, root_cause, target_var,
        max_path_len=len(selected_cols)
    )
    
    top_paths = select_top_paths(all_paths, top_k=top_k)
    
    strongest_path = top_paths[0] if top_paths else None
    
    return {
        "pcmci_edges": pcmci_edges,
        "propagation_edges": propagation_edges,
        "all_paths": all_paths,
        "top_paths": top_paths,
        "strongest_path": strongest_path
    }
