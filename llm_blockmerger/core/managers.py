import os, json

def initialize_managers(notebook_paths):
    from load.block_loading import CodeBlocksManager
    managers = []
    for notebook_path in notebook_paths:
        managers.append(CodeBlocksManager())
        managers[-1].preprocess_notebook(*load_notebooks(notebook_path))
    return managers

def concatenate_managers(block_managers):
    labels, blocks, sources, variables, var_descriptions = [], [], [], [], []
    for block_manager in block_managers:
        labels.extend(block_manager.labels)
        blocks.extend(block_manager.blocks)
        variables.extend(block_manager.variables)
        sources.extend(block_manager.sources for _ in range(len(block_manager)))
        var_descriptions.extend(block_manager.var_descriptions)
    return blocks, labels, variables, var_descriptions, sources

def concatenate_block(block):
    return '\n'.join(block) + '\n'

def load_notebooks(nb_paths):
    if not isinstance(nb_paths, list): return os.path.basename(nb_paths), json.load(open(nb_paths, 'r'))
    return [(os.path.basename(path), json.load(open(path, 'r'))) for path in nb_paths]
