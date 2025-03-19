import numpy as np
from llm_blockmerger.load.block_loading import CodeBlocksManager
from llm_blockmerger.core.utils import remove_common_words, embedding_projection

def linear_string_merge(embedding_model, vector_db, specification, max_iterations=10, replacement='UNKNOWN'):
    merge_block_manager = CodeBlocksManager()
    for _ in range(max_iterations):
        # Break condition: Exit if the empty string is empty
        if specification in ['', ' ']:
            return merge_block_manager

        spec_embedding = embedding_model.encode_strings(specification)
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
    return embedding_model, merge_block_manager

def linear_embedding_merge(embedding_model, vector_db, specification, max_iterations=10, norm_threshold=0.1):
    merge_block_manager = CodeBlocksManager()
    current_embedding = embedding_model.encode_strings(specification)[0]

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
        neighbor_embedding = embedding_model.encode_strings(nearest_doc.label)[0]
        projection = embedding_projection(current_embedding, neighbor_embedding)

        # Break condition: Exit when current and neighbor embedding vectors are perpendicular
        if np.linalg.norm(projection) < norm_threshold:
            break
        current_embedding = current_embedding - projection
    return embedding_model, merge_block_manager


