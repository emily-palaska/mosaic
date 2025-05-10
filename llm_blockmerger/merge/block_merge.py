import numpy as np
from llm_blockmerger.load import CodeBlocksManager
from llm_blockmerger.core import (remove_common_words, load_double_encoded_json,
                                  embedding_projection, LLM, find_block_order)
from llm_blockmerger.merge import merge_variables
from llm_blockmerger.store import BlockMergerVectorDB

def linear_string_merge(embedding_model: LLM, vector_db: BlockMergerVectorDB,
                        specification: str, max_iterations=10, replacement='UNKNOWN', var_merge=True):
    merge_block_manager = CodeBlocksManager()

    for _ in range(max_iterations):
        if specification in ['', ' ']: return merge_block_manager # Break condition: Empty specification

        spec_embedding = embedding_model.encode_strings(specification)
        nearest_neighbors = vector_db.read(spec_embedding, limit=1)
        if not nearest_neighbors: break # Break condition: No neighbors

        nearest_doc = nearest_neighbors[0]
        merge_block_manager.append_doc(nearest_doc)
        nearest_doc_label = load_double_encoded_json(nearest_doc.blockdata)['label']
        new_specification = remove_common_words(original=specification, to_remove=''.join(nearest_doc_label),
                                                replacement=replacement)
        if new_specification == specification: break # Break condition: Unchanged specification
        specification = new_specification

    merge_block_manager.rearrange(find_block_order(
        blocks=merge_block_manager.blocks,
        var_dicts=merge_block_manager.variable_dictionaries
    ))
    return merge_variables(embedding_model, merge_block_manager) if var_merge else merge_block_manager

def linear_embedding_merge(embedding_model: LLM, vector_db: BlockMergerVectorDB,
                           specification: str, max_iterations=10, norm_threshold=0.2, var_merge=True):
    merge_block_manager = CodeBlocksManager()
    current_embedding = embedding_model.encode_strings(specification)[0]

    for _ in range(max_iterations):
        if np.linalg.norm(current_embedding) < norm_threshold: break # Break condition: Embedding norm below threshold

        nearest_neighbors = vector_db.read(current_embedding, limit=1)
        if not nearest_neighbors: break # Break condition: No neighbors

        merge_block_manager.append_doc(nearest_neighbors[0])
        nearest_doc_label = load_double_encoded_json(nearest_neighbors[0].blockdata)['label']
        neighbor_embedding = embedding_model.encode_strings(nearest_doc_label)[0]
        projection = embedding_projection(current_embedding, neighbor_embedding)
        if np.linalg.norm(projection) < norm_threshold: break # Break condition: Perpendicular embeddings

        current_embedding = current_embedding - projection
    merge_block_manager.rearrange(find_block_order(
        blocks=merge_block_manager.blocks,
        var_dicts=merge_block_manager.variable_dictionaries
    ))
    return merge_variables(embedding_model, merge_block_manager) if var_merge else merge_block_manager



