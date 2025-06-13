from llm_blockmerger.load import BlockManager
from llm_blockmerger.core import (remove_common_words, encoded_json,
                                  projection, LLM)
from llm_blockmerger.merge.variable_merge import merge_variables
from llm_blockmerger.merge.block_reorder import  find_block_order, cumulative_io_split
from llm_blockmerger.store import BlockDB
from torch import norm, tensor

def linear_string_merge(embedding_model: LLM, vector_db: BlockDB, specification: str,
                        max_rep=2, max_it=10, replacement='UNKNOWN', var_merge=True):
    merge_block_manager = BlockManager()
    neighbor_ids = []

    import textwrap
    for _ in range(max_it):
        print(textwrap.fill(f'Current specification: {specification}', 100))
        if specification in ['', ' ']: return merge_block_manager # Break condition: Empty specification

        spec_embedding = embedding_model.encode_strings(specification)
        nearest_neighbor = check_repetitions(max_rep, neighbor_ids, vector_db.read(spec_embedding, limit=3))
        if nearest_neighbor is None: break  # Break condition: No valid neighbors

        neighbor_ids.append(nearest_neighbor.id)
        nearest_doc_label = encoded_json(nearest_neighbor.blockdata)['label']
        new_specification = remove_common_words(original=specification, to_remove=''.join(nearest_doc_label),
                                                replacement=replacement)
        if new_specification == specification: break # Break condition: Unchanged specification

        merge_block_manager.append_doc(nearest_neighbor)
        specification = new_specification

    merge_block_manager.rearrange(find_block_order(cumulative_io_split(merge_block_manager)))
    return merge_variables(embedding_model, merge_block_manager) if var_merge else merge_block_manager


def linear_embedding_merge(embedding_model: LLM, vector_db: BlockDB, specification: str,
                           k=0.9, l=1.4, max_it=10, norm_threshold=0.05, var_merge=True):
    merge_block_manager = BlockManager()
    search_embedding = tensor(embedding_model.encode_strings(specification)[0])
    specification_embedding = tensor(embedding_model.encode_strings(specification)[0])
    information = specification_embedding.norm().item()

    for _ in range(max_it):
        print(f'Search embedding: {search_embedding.norm().item(): .2f}, Information: {information: .2f}')
        if information < norm_threshold: break # Break condition: Embedding norm below the norm threshold

        nearest_neighbor = vector_db.read(search_embedding, limit=1)[0]
        if nearest_neighbor is None: break  # Break condition: No neighbors

        neighbor_embedding = nearest_neighbor.embedding
        neighbor_projection = projection(neighbor_embedding, search_embedding)
        info_projection = projection(specification_embedding, neighbor_embedding)
        if norm(neighbor_projection) < norm_threshold: break  # Break condition: Perpendicular embeddings

        merge_block_manager.append_doc(nearest_neighbor)

        search_embedding = l * neighbor_projection - search_embedding
        search_embedding /= search_embedding.norm()
        information -= k * info_projection.norm().item()

    merge_block_manager.rearrange(find_block_order(cumulative_io_split(merge_block_manager)))
    return merge_variables(embedding_model, merge_block_manager) if var_merge else merge_block_manager


def check_repetitions(max_rep:int, neighbor_ids:list, top_nearest_neighbors:list):
    for neighbor in top_nearest_neighbors:
        if neighbor_ids.count(neighbor.id) < max_rep:
            return neighbor
    return None