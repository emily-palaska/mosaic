import numpy as np
from llm_blockmerger.encoding.embedding_model import encode_labels
from llm_blockmerger.loading.blockloading import _concatenate_block, CodeBlocksManager
import textwrap

def remove_common_words(original: str, to_remove: str, replacement='UNKNOWN') -> str:
    original = original.replace('\n', ' ')
    original_words = original.split()
    remove_words = set(word.lower() for word in to_remove.split())

    replaced_words = [
        replacement if word.lower() in remove_words else word
        for word in original_words
    ]
    return ' '.join(replaced_words)

def linear_string_merge(embedding_model, vector_db, specification, max_iterations=10, replacement='UNKNOWN'):
    merge_block_manager = CodeBlocksManager()
    for _ in range(max_iterations):
        # Break condition: Exit if the empty string is empty
        if specification in ['', ' ']:
            return merge_block_manager

        spec_embedding = encode_labels(embedding_model, [specification])
        nearest_neighbors = vector_db.read(spec_embedding, limit=1)

        # Break condition: Exit if no neighbors are found
        if not nearest_neighbors:
            break

        nearest_doc = nearest_neighbors[0]
        merge_block_manager.append_doc(nearest_doc)
        new_specification = remove_common_words(specification, ''.join(nearest_doc.label), replacement=replacement)

        # Break condition: Exit if specification remains the same
        if new_specification == specification:
            break
        specification = new_specification
    return merge_block_manager

def embedding_projection(current_embedding, neighbor_embedding):
    if np.all(current_embedding == 0):
        return None
    inner_product = np.dot(neighbor_embedding, current_embedding) / np.dot(current_embedding, current_embedding)
    return inner_product * current_embedding

def linear_embedding_merge(embedding_model, vector_db, specification, max_iterations=10, norm_threshold=0.1):
    merge_block_manager = CodeBlocksManager()
    current_embedding = encode_labels(embedding_model, [specification])[0]

    for _ in range(max_iterations):
        # Break condition: Exit if embedding norm below threshold
        if np.linalg.norm(current_embedding) < norm_threshold:
            break

        # Break condition: Exit if no neighbors are found
        nearest_neighbors = vector_db.read(current_embedding, limit=1)
        if not nearest_neighbors:
            break

        nearest_doc = nearest_neighbors[0]
        merge_block_manager.append_doc(nearest_doc)
        neighbor_embedding = encode_labels(embedding_model, [nearest_doc.label])[0]
        projection = embedding_projection(current_embedding, neighbor_embedding)

        # Break condition: Exit when current and neighbor embedding vectors are perpendicular
        if np.linalg.norm(projection) < norm_threshold:
            break
        current_embedding = current_embedding - projection
    return merge_block_manager

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
        print(_concatenate_block(block))
        print("VARIABLES:")
        for var, desc in zip(variables, var_descriptions):
            print(f'{var}:{textwrap.fill(desc,100)}')

    print("\n" + "=" * 60)
