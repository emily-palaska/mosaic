import numpy as np
from llm_blockmerger.load import CodeBlocksManager
from llm_blockmerger.core import remove_common_words, load_double_encoded_json, embedding_projection, LLM
from llm_blockmerger.merge import merge_variables
from llm_blockmerger.store import BlockMergerVectorDB

def linear_string_merge(embedding_model: LLM,
                        vector_db: BlockMergerVectorDB,
                        specification: str,
                        max_iterations=10,
                        replacement='UNKNOWN',
                        variable_merge=True):
    merge_block_manager = CodeBlocksManager(variable_dictionaries=dict())
    for _ in range(max_iterations):
        # Break condition: Exit if the specification is empty
        if specification in ['', ' ']: return merge_block_manager

        spec_embedding = embedding_model.encode_strings(specification)
        nearest_neighbors = vector_db.read(spec_embedding, limit=1)

        # Break condition: Exit if no neighbors are found
        if not nearest_neighbors: break

        nearest_doc = nearest_neighbors[0]
        merge_block_manager.append_doc(nearest_doc)
        nearest_doc_label = load_double_encoded_json(nearest_doc.blockdata)['label']
        new_specification = remove_common_words(original=specification,
                                                to_remove=''.join(nearest_doc_label),
                                                replacement=replacement)

        # Break condition: Exit if specification remains the same
        if new_specification == specification: break
        specification = new_specification
    return merge_variables(embedding_model, merge_block_manager) if variable_merge else merge_block_manager

def linear_embedding_merge(embedding_model: LLM,
                           vector_db: BlockMergerVectorDB,
                           specification: str,
                           max_iterations=10,
                           norm_threshold=0.5,
                           variable_merge=True):
    merge_block_manager = CodeBlocksManager(variable_dictionaries=dict())
    current_embedding = embedding_model.encode_strings(specification)[0]

    for _ in range(max_iterations):
        # Break condition: Exit if embedding norm below threshold
        if np.linalg.norm(current_embedding) < norm_threshold: break

        # Break condition: Exit if no neighbors are found
        nearest_neighbors = vector_db.read(current_embedding, limit=1)
        if not nearest_neighbors: break

        nearest_doc = nearest_neighbors[0]
        merge_block_manager.append_doc(nearest_doc)
        nearest_doc_label = load_double_encoded_json(nearest_doc.blockdata)['label']
        neighbor_embedding = embedding_model.encode_strings(nearest_doc_label)[0]
        projection = embedding_projection(current_embedding, neighbor_embedding)

        # Break condition: Exit when current and neighbor embedding vectors are perpendicular
        if np.linalg.norm(projection) < norm_threshold: break
        current_embedding = current_embedding - projection
    return merge_variables(embedding_model, merge_block_manager) if variable_merge else merge_block_manager



