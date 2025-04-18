from llm_blockmerger.core.embeddings import compute_embedding_similarity
from llm_blockmerger.load.managers import CodeBlocksManager
from collections import defaultdict

def _build_similarity_graph(variables, similarity_matrix, threshold):
    graph = defaultdict(set)
    n = len(variables)
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i][j] > threshold:
                graph[variables[i]].add(variables[j])
                graph[variables[j]].add(variables[i])
    return graph

def _find_connected_components(graph):
    visited = set()
    components = []

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

def _find_pair_merges(embedding_model, variable_dictionaries, threshold=0.9):
    flat_variables = [v for block_dict in variable_dictionaries for v in block_dict]
    flat_descriptions = [d for block_dict in variable_dictionaries for d in block_dict.values()]

    similarity_matrix = compute_embedding_similarity(embedding_model.encode_strings(flat_descriptions),)
    components = [
        [flat_variables[i], flat_variables[j]]
                for i in range(len(flat_variables))
                for j in range(i + 1, len(flat_variables))
                if similarity_matrix[i][j] > threshold
    ]
    return components

def _find_group_merges(embedding_model, variable_dictionaries, threshold=0.9):
    flat_variables = [v for block_dict in variable_dictionaries for v in block_dict]
    flat_descriptions = [d for block_dict in variable_dictionaries for d in block_dict.values()]

    embeddings = embedding_model.encode_strings(flat_descriptions)
    similarity_matrix = compute_embedding_similarity(embeddings)

    graph = _build_similarity_graph(flat_variables, similarity_matrix, threshold)

    return _find_connected_components(graph)

def _refactor_dictionaries(components, variable_dictionaries):
    to_remove = [v for group in components for v in group[1:]]

    for block_dictionary in variable_dictionaries:
        for v in to_remove:
            if v in block_dictionary:
                del block_dictionary[v]

def _replace_variables(block, old_var, new_var):
    modified_block = []
    for line in block:
        import re
        modified_line = re.sub(r'\b' + re.escape(old_var) + r'\b', new_var, line)
        modified_block.append("".join(modified_line))
    return modified_block

def merge_variables(embedding_model, block_manager, threshold=0.9, find_merges=_find_group_merges):
    if find_merges not in [_find_group_merges, _find_pair_merges]:
        raise TypeError("Incorrect type of find_variables_function")
    blocks, _, variable_dictionaries, _ = block_manager.unzip()

    components = find_merges(embedding_model, variable_dictionaries, threshold)
    _refactor_dictionaries(components, variable_dictionaries)

    for group in components:
        for idg in range(1, len(group)):
            for idx, block in enumerate(blocks):
                blocks[idx] = _replace_variables(block, group[idg], group[0])

    block_manager.set(blocks=blocks, variable_dictionaries=variable_dictionaries)
    return block_manager

# todo make tests out of this execution scenario
def main():
    demo_blocks = [['x = 10', 'y = 10', 'z = 2'], ['a = 1', 'print(a)']]
    demo_variable_dictionaries = [
        {
            'x': 'Variable set to ten',
            'y': 'A variable set to ten.',
            'z': 'Variable set to two.'
        },
        {
            'a': 'Variable set to one and printed.'
        }
    ]

    from llm_blockmerger.core.models import LLM
    embedding_model = LLM(task='embedding')

    manager = CodeBlocksManager(
        blocks=demo_blocks,
        variable_dictionaries=demo_variable_dictionaries
    )

    manager = merge_variables(embedding_model, manager)
    print(manager.blocks)

    for block_dict in manager.variable_dictionaries:
        for v, d in block_dict.items():
            print(f'{v}: {d}')
        print('-'*40)

if __name__ == '__main__':
    main()
