from collections import defaultdict
from mosaic.core import norm_cos_sim
from mosaic.core.utils import regular_replace

def _connected_components(graph):
    visited, components = set(), []

    def dfs(n, g):
        visited.add(n)
        g.append(n)
        for neighbor in graph[n]:
            if neighbor not in visited:
                dfs(neighbor, g)

    for node in graph:
        if node not in visited:
            group = []
            dfs(node, group)
            components.append(group)

    return components


def _pair_merges(model, var_dicts, t=0.9):
    flat_vars = [v for block_dict in var_dicts for v in block_dict]
    flat_descs = [d for block_dict in var_dicts for d in block_dict.values()]

    sim_mat = norm_cos_sim(model.encode(flat_descs))
    components = [
        [flat_vars[i], flat_vars[j]]
                for i in range(len(flat_vars))
                for j in range(i + 1, len(flat_vars))
                if sim_mat[i][j] > t
    ]
    return components


def _sim_graph(vars, sim_mat, t):
    graph = defaultdict(set)
    for i in range(len(vars)):
        for j in range(i + 1, len(vars)):
            if sim_mat[i][j] > t:
                graph[vars[i]].add(vars[j])
                graph[vars[j]].add(vars[i])
    return graph


def _group_merges(model, var_dicts, t=0.9):
    if isinstance(var_dicts, dict):
        flat_vars = list(var_dicts.keys())
        flat_descs = list(var_dicts.values())
    elif isinstance(var_dicts, list):
        flat_vars = [v for block_dict in var_dicts for v in block_dict]
        flat_descs = [d for block_dict in var_dicts for d in block_dict.values()]
    else:
        raise TypeError(f'Incorrect type {type(var_dicts)} for var_dicts')

    embeddings = model.encode(flat_descs)
    sim_mat = norm_cos_sim(embeddings)
    graph = _sim_graph(flat_vars, sim_mat, t)
    return _connected_components(graph)


def _merge_dicts(components, var_dicts):
    if not isinstance(var_dicts, list): var_dicts = [var_dicts]
    to_remove = [v for group in components for v in group[1:]]

    for block_dict in var_dicts:
        for v in to_remove:
            if v in block_dict:
                del block_dict[v]


def merge_variables(model, manager, t=0.9, merge_type=_group_merges):
    assert merge_type in [_group_merges, _pair_merges], f"Incorrect merge type: {merge_type}"
    blocks, _, var_dicts, _ = manager.unzip()
    components = merge_type(model, var_dicts, t)
    _merge_dicts(components, var_dicts)

    for group in components:
        for idg in range(1, len(group)):
            for idx, block in enumerate(blocks):
                blocks[idx] = regular_replace(block, group[idg], group[0])

    manager.set(blocks=blocks, var_dicts=var_dicts)
    return manager
