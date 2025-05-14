import numpy as np
from llm_blockmerger.load import CodeBlocksManager
from llm_blockmerger.core import (remove_common_words, load_double_encoded_json,
                                  embedding_projection, LLM, ast_io_split)
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

    io_splits = [
        ast_io_split(script=block, variables=var_dict)
        for block, var_dict in zip(merge_block_manager.blocks, merge_block_manager.variable_dictionaries)
    ]
    merge_block_manager.rearrange(find_block_order(io_splits))
    return merge_variables(embedding_model, merge_block_manager) if var_merge else merge_block_manager

def linear_embedding_merge(embedding_model: LLM, vector_db: BlockMergerVectorDB,
                           specification: str, max_iterations=10, norm_threshold=0.2, var_merge=True):
    merge_block_manager = CodeBlocksManager()
    current_embedding = embedding_model.encode_strings(specification)[0]
    scale = 2.0
    for _ in range(max_iterations):
        if np.linalg.norm(current_embedding) < norm_threshold: break # Break condition: Embedding norm below threshold

        nearest_neighbors = vector_db.read(current_embedding, limit=1)
        if not nearest_neighbors: break # Break condition: No neighbors

        merge_block_manager.append_doc(nearest_neighbors[0])
        nearest_doc_label = load_double_encoded_json(nearest_neighbors[0].blockdata)['label']
        neighbor_embedding = embedding_model.encode_strings(nearest_doc_label)[0]
        projection = embedding_projection(current_embedding, neighbor_embedding)
        if np.linalg.norm(projection) < norm_threshold: break # Break condition: Perpendicular embeddings

        #norma = np.linalg.norm(current_embedding)
        current_embedding = current_embedding - scale * projection
        #nea_norma = np.linalg.norm(current_embedding)
        #current_embedding = norma * current_embedding / nea_norma

    io_splits = [
        ast_io_split(script=block, variables=var_dict)
            for block, var_dict in zip(merge_block_manager.blocks, merge_block_manager.variable_dictionaries)
    ]
    merge_block_manager.rearrange(find_block_order(io_splits))
    return merge_variables(embedding_model, merge_block_manager) if var_merge else merge_block_manager

def find_block_order(io_splits: list):
    all_outputs = set()
    all_outputs.add(output_var for io_split in io_splits for output_var in io_split['output'])

    order, used_outputs = [], set()
    remaining = set(range(len(io_splits)))
    while remaining:
        scheduled = False
        for i in list(remaining):
            if io_splits[i]['input'].issubset(used_outputs):
                order.append(i)
                used_outputs |= io_splits[i]['output']
                remaining.remove(i)
                scheduled = True
                break
        if not scheduled: # Fallback: pick one block to reduce deadlock
            i = remaining.pop()
            order.append(i)
            used_outputs |= var_splits[i]['output']

    return order

