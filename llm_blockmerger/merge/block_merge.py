import numpy as np
from llm_blockmerger.load import CodeBlocksManager
from llm_blockmerger.core import (remove_common_words, load_double_encoded_json,
                                  embedding_projection, LLM, ast_io_split)
from llm_blockmerger.merge import merge_variables
from llm_blockmerger.store import BlockMergerVectorDB

def linear_string_merge(embedding_model: LLM, vector_db: BlockMergerVectorDB, specification: str,
                        max_rep=2, max_it=10, replacement='UNKNOWN', var_merge=True):
    merge_block_manager = CodeBlocksManager()
    neighbor_ids = []

    import textwrap
    for _ in range(max_it):
        print(textwrap.fill(f'Current specification: {specification}', 100))
        if specification in ['', ' ']: return merge_block_manager # Break condition: Empty specification

        spec_embedding = embedding_model.encode_strings(specification)
        top_nearest_neighbors = vector_db.read(spec_embedding, limit=3)
        nearest_neighbor = None

        for i in range(3):
            if neighbor_ids.count(top_nearest_neighbors[i].id) < max_rep:
                neighbor_ids.append(top_nearest_neighbors[i].id)
                nearest_neighbor = top_nearest_neighbors[i]
                break
        if nearest_neighbor is None: break  # Break condition: No neighbors

        merge_block_manager.append_doc(nearest_neighbor)
        nearest_doc_label = load_double_encoded_json(nearest_neighbor.blockdata)['label']
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

def linear_embedding_merge(embedding_model: LLM, vector_db: BlockMergerVectorDB, specification: str,
                           scale=1.0, max_rep=2, max_it=10, norm_threshold=0.1, var_merge=True):
    merge_block_manager = CodeBlocksManager()
    current_embedding = embedding_model.encode_strings(specification)[0]
    neighbor_ids = []

    for _ in range(max_it):
        print(f'Current embedding: {np.linalg.norm(current_embedding)}')
        if np.linalg.norm(current_embedding) < norm_threshold: break # Break condition: Embedding norm below threshold

        top_nearest_neighbors = vector_db.read(current_embedding, limit=3)
        nearest_neighbor = None

        for i in range(3):
            if neighbor_ids.count(top_nearest_neighbors[i].id) < max_rep:
                neighbor_ids.append(top_nearest_neighbors[i].id)
                nearest_neighbor = top_nearest_neighbors[i]
                break
        if nearest_neighbor is None: break  # Break condition: No neighbors

        merge_block_manager.append_doc(nearest_neighbor)
        nearest_doc_label = load_double_encoded_json(nearest_neighbor.blockdata)['label']
        neighbor_embedding = embedding_model.encode_strings(nearest_doc_label)[0]
        projection = embedding_projection(current_embedding, neighbor_embedding)
        if np.linalg.norm(projection) < norm_threshold: break # Break condition: Perpendicular embeddings

        current_embedding = current_embedding - scale * projection

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
            used_outputs |= io_splits[i]['output']

    return order

