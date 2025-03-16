from llm_blockmerger.encoding.embedding_model import initialize_model, encode_labels, compute_similarity
from llm_blockmerger.loading.blockloading import CodeBlocksManager

def merge_variables(block_manager, threshold=0.9):
    blocks, labels, variables, var_descriptions, sources = block_manager.unzip()
    variables_to_merge, variables_to_remove, descriptions_to_remove = _find_variables_to_remove(*_find_variables_to_merge(variables, var_descriptions, threshold))

    for var, desc in zip(variables, var_descriptions):
        var[:] = [v for v in var if v not in variables_to_remove]
        desc[:] = [d for d in desc if d not in descriptions_to_remove]

    for old_var, new_var in variables_to_merge:
        for idx, block in enumerate(blocks):
            blocks[idx] = _replace_variables(block, old_var, new_var)

    block_manager.set(blocks=blocks, variables=variables, var_descriptions=var_descriptions)
    return block_manager


def _find_variables_to_merge(variables, var_descriptions, threshold=0.9):
    flat_variables = [item for sublist in variables for item in sublist]
    flat_descriptions = [item for sublist in var_descriptions for item in sublist]

    similarity_matrix = compute_similarity(encode_labels(initialize_model(), flat_descriptions))
    variables_to_merge = [(flat_variables[i], flat_variables[j])
                for i in range(len(flat_variables))
                for j in range(i + 1, len(flat_variables))
                if similarity_matrix[i][j] > threshold]
    return variables_to_merge, flat_variables, flat_descriptions


def _find_variables_to_remove(variables_to_merge, flat_variables, flat_descriptions):
    variables_to_remove, descriptions_to_remove = set(), set()

    for pair in variables_to_merge:
        variables_to_remove.add(pair[0])
        idx = flat_variables.index(pair[0])
        descriptions_to_remove.add(flat_descriptions[idx])
    return variables_to_merge, variables_to_remove, descriptions_to_remove


def _replace_variables(block, old_var, new_var):
    modified_block = []
    for line in block:
        import re
        modified_line = re.sub(r'\b' + re.escape(old_var) + r'\b', new_var, line)
        modified_block.append("".join(modified_line))
    return modified_block

# todo make tests out of this execution scenario
def main():
    demo_blocks = [['x = 10', 'y = 10', 'z = 2'], ['a = 1', 'print(a)']]
    demo_variables = [['x', 'y', 'z'], ['a']]
    demo_var_descriptions = [
        [
            'Variable set to ten',
            'A variable set to ten.',
            'Variable set to two.'
        ],
        [
            'Variable set to one and printed.',
        ]
    ]

    manager = CodeBlocksManager(
        blocks=demo_blocks,
        variables=demo_variables,
        var_descriptions=demo_var_descriptions
    )
    manager = merge_variables(manager)
    print(manager.blocks)
    print(manager.variables)
    print(manager.var_descriptions)

if __name__ == '__main__':
    main()
