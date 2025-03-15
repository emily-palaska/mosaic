from torch.autograd import variable

from llm_blockmerger.encoding.embedding_model import initialize_model, encode_labels, compute_similarity
from llm_blockmerger.loading.blockloading import CodeBlocksManager


def merge_variables(block_manager, threshold=0.9):
    blocks, labels, variables, var_descriptions, sources = block_manager.unzip()

    embedding_model = initialize_model()
    encoded_descriptions = encode_labels(embedding_model, var_descriptions)
    similarity_matrix = compute_similarity(encoded_descriptions)

    to_merge = []
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            if similarity_matrix[i][j] > threshold:
                to_merge.append((variables[i], variables[j]))
                variables.remove(variables[i])

    for old_var, new_var in to_merge:
        for idx, block in enumerate(blocks):
            blocks[idx] = _replace_variables(block, old_var, new_var)

    block_manager.blocks = blocks
    return block_manager

def _replace_variables(block, old_var, new_var):
    modified_block = []
    for line in block:
        # Use regex to replace whole words only, avoiding partial matches
        import re
        modified_line = re.sub(r'\b' + re.escape(old_var) + r'\b', new_var, line)
        modified_block.append(modified_line)
    return "".join(modified_block)


if __name__ == '__main__':
    # todo refactor for many blocks
    demo_blocks = ['x = 10', 'y = 10', 'z = 2']
    demo_variables = ['x', 'y', 'z']
    demo_var_descriptions = [
        'Variable set to ten',
        'A variable set to ten.',
        'Variable set to two.'
    ]
    manager = CodeBlocksManager(
        blocks=demo_blocks,
        variables=demo_variables,
        var_descriptions=demo_var_descriptions
    )
    manager = merge_variables(manager)
    print(manager.blocks)
    print(manager.variables)