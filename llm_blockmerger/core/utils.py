import ast, os, json, textwrap
from sentence_transformers import util
import matplotlib.pyplot as plt
import numpy as np

def remove_common_words(original: str, to_remove: str, replacement='UNKNOWN') -> str:
    original = original.replace('\n', ' ')
    original_words = original.split()
    remove_words = set(word.lower() for word in to_remove.split())

    replaced_words = [
        replacement if word.lower() in remove_words else word
        for word in original_words
    ]
    return ' '.join(replaced_words)

def embedding_projection(current_embedding, neighbor_embedding):
    if np.all(current_embedding == 0):
        return None
    inner_product = np.dot(neighbor_embedding, current_embedding) / np.dot(current_embedding, current_embedding)
    return inner_product * current_embedding

def compute_embedding_similarity(embeddings):
    return util.pytorch_cos_sim(embeddings, embeddings)

def plot_similarity_matrix(similarity_matrix, save_path='../plots/similarity_matrix.png'):
    plt.figure(figsize=(12, 12))
    plt.imshow(similarity_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Similarity')

    # Annotate the matrix with similarity values - ONLY for small dimensions due to visibility
    if similarity_matrix.shape[0] <= 20:
        for i in range(similarity_matrix.shape[0]):
            for j in range(similarity_matrix.shape[1]):
                plt.text(j, i, f"{similarity_matrix[i, j]:.2f}",
                         ha="center", va="center", color="black")

    plt.title("Similarity Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

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
        sources.extend(block_manager.sources for _ in range(len(block_manager.blocks)))
        var_descriptions.extend(block_manager.var_descriptions)
    return blocks, labels, variables, var_descriptions, sources

def concatenate_block(block):
    return '\n'.join(block) + '\n'

def load_notebooks(nb_paths):
    if not isinstance(nb_paths, list): return os.path.basename(nb_paths), json.load(open(nb_paths, 'r'))
    return [(os.path.basename(path), json.load(open(path, 'r'))) for path in nb_paths]

def ast_extraction(script=''):
    tree = ast.parse(script)
    variables = set()

    def handle_function(curr_node):
        for arg in curr_node.args.args:
            variables.add(arg.arg)

    def handle_loop(curr_node):
        if isinstance(curr_node.target, ast.Name):
            variables.add(curr_node.target.id)
        elif isinstance(curr_node.target, ast.Tuple):
            handle_tuple(curr_node)

    def handle_tuple(curr_node):
        for target in curr_node.targets:
            if isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        variables.add(elt.id)

    def visit_node(curr_node):
        match curr_node:
            case ast.Name(ctx=ast.Store()):
                variables.add(curr_node.id)
            case ast.FunctionDef():
                handle_function(curr_node)
            case ast.For():
                handle_loop(curr_node)
            case ast.Assign():
                handle_tuple(curr_node)

    for node in ast.walk(tree):
        visit_node(node)

    return sorted(list(variables))

def print_merge_result(specification, block_manager):
    print("\n" + "=" * 60)
    print(' ' * 23 + "MERGE RESULT")
    print("=" * 60)

    print("\nSpecification (Input to Merging Mechanism):")
    print(textwrap.indent(specification, "    "))
    print("=" * 60)

    blocks, labels, variables, var_descriptions, sources = block_manager.unzip()
    for i, (label, block, source) in enumerate(zip(labels, blocks, sources), 1):
        print("\n" + "-" * 60)
        print(f"SOURCE: {source}")
        print(textwrap.fill(label,100))
        print("CODE:")
        print(concatenate_block(block))
        print("VARIABLES:")
        for var, desc in zip(variables, var_descriptions):
            print(f'{var}:{textwrap.fill(desc,100)}')

    print("\n" + "=" * 60)