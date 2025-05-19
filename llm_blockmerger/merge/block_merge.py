from numpy.linalg import norm
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
        nearest_neighbor = check_repetitions(max_rep, neighbor_ids, vector_db.read(spec_embedding, limit=3))
        if nearest_neighbor is None: break  # Break condition: No valid neighbors

        neighbor_ids.append(nearest_neighbor.id)
        nearest_doc_label = load_double_encoded_json(nearest_neighbor.blockdata)['label']
        new_specification = remove_common_words(original=specification, to_remove=''.join(nearest_doc_label),
                                                replacement=replacement)
        if new_specification == specification: break # Break condition: Unchanged specification

        merge_block_manager.append_doc(nearest_neighbor)
        specification = new_specification

    merge_block_manager.rearrange(find_block_order(cumulative_io_split(merge_block_manager)))
    return merge_variables(embedding_model, merge_block_manager) if var_merge else merge_block_manager


def linear_embedding_merge(embedding_model: LLM, vector_db: BlockMergerVectorDB, specification: str,
                           l=0.2, max_rep=1, max_it=10, norm_threshold=0.1, var_merge=True):
    merge_block_manager = CodeBlocksManager()
    search_embedding = embedding_model.encode_strings(specification)[0]
    info_embedding = embedding_model.encode_strings(specification)[0]
    neighbor_ids = []

    for _ in range(max_it):
        print(f'Search embedding: {norm(search_embedding)}, Information: {norm(info_embedding)}')
        if norm(info_embedding) < norm_threshold: break # Break condition: Embedding norm below threshold

        nearest_neighbor = check_repetitions(max_rep, neighbor_ids, vector_db.read(search_embedding, limit=3))
        if nearest_neighbor is None: break  # Break condition: No neighbors

        neighbor_ids.append(nearest_neighbor.id)
        nearest_doc_label = load_double_encoded_json(nearest_neighbor.blockdata)['label']
        neighbor_embedding = embedding_model.encode_strings(nearest_doc_label)[0]

        search_projection = embedding_projection(search_embedding, neighbor_embedding)
        #if norm(search_projection) < norm_threshold: break  # Break condition: Perpendicular embeddings
        #if norm(search_projection) > 0.98: break # Break condition: Identical embeddings

        info_projection = embedding_projection(info_embedding, neighbor_embedding)
        #if norm(info_projection) < norm_threshold: break  # Break condition: Perpendicular embeddings
        #if norm(info_projection) > 0.98: break  # Break condition: Identical embeddings

        merge_block_manager.append_doc(nearest_neighbor)
        search_embedding = search_embedding - search_projection + l*neighbor_embedding
        search_embedding /= norm(search_embedding)
        info_embedding = info_embedding - l*info_projection

    merge_block_manager.rearrange(find_block_order(cumulative_io_split(merge_block_manager)))
    return merge_variables(embedding_model, merge_block_manager) if var_merge else merge_block_manager


def check_repetitions(max_rep:int, neighbor_ids:list, top_nearest_neighbors:list):
    for neighbor in top_nearest_neighbors:
        if neighbor_ids.count(neighbor.id) < max_rep:
            return neighbor
    return None


def cumulative_io_split(manager:CodeBlocksManager):
    return [
        ast_io_split(script=block, variables=var_dict)
            for block, var_dict in zip(manager.blocks, manager.variable_dictionaries)
    ]


def find_block_order(io_splits: list):
    all_outputs = set()
    all_outputs.add(output_var for io_split in io_splits for output_var in io_split['output'])

    order, used_outputs = [], set()
    remaining = set(range(len(io_splits)))
    while remaining:
        min_violations, min_index = float('inf'), 0
        for i in list(remaining):
            violations = len(io_splits[i]['input'] - used_outputs)
            if violations < min_violations:
                min_violations = violations
                min_index = i

        order.append(min_index)
        used_outputs |= io_splits[min_index]['output']
        remaining.remove(min_index)
    return order