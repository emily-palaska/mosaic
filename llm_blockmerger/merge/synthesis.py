from llm_blockmerger.core.embeddings import pivot_rotation
from llm_blockmerger.load import BlockManager
from llm_blockmerger.core import remove_common_words, encoded_json, projection, LLM, ast_io_split, best_combination
from llm_blockmerger.merge.merger import merge_variables
from llm_blockmerger.merge.order import io_order
from llm_blockmerger.learn import MLP
from llm_blockmerger.store import BlockDB

from torch import tensor, Tensor


def check_repetitions(max_rep:int, ids:list, top_nn:list):
    for neighbor in top_nn:
        if ids.count(neighbor.id) < max_rep:
            return neighbor
    return None

def string_synthesis(model:LLM, db:BlockDB, query:str, max_rep=2, repl:str='[UNK]',
                     var:bool=True, max_it:int|None=None, mlp:MLP|None=None):
    synthesis, ids = BlockManager(), []

    s = tensor(model.encode(query))
    if mlp: s = mlp(s)

    if max_it is None: max_it = s.shape[0]
    for _ in range(max_it):
        if query in ['', ' ']: return synthesis # Break condition: Empty specification

        nn = check_repetitions(max_rep, ids, db.read(s, limit=3))
        if nn is None: break  # Break condition: No valid neighbors

        ids.append(nn.id)
        nn_label = encoded_json(nn.blockdata)['label']
        new_query = remove_common_words(og=query, rem=''.join(nn_label), repl=repl)
        if new_query == query: break # Break condition: Unchanged specification

        synthesis.append_doc(nn)
        query = new_query
        s = tensor(model.encode(query))
        if mlp: s = mlp(s)

    synthesis.rearrange(io_order(ast_io_split(synthesis)))
    return merge_variables(model, synthesis) if var else synthesis


def embedding_synthesis(model:LLM, db:BlockDB, query:str, k:float=0.9, l:float=1.4, t:float=0.05,
                        max_it:int|None=None, mlp:MLP|None=None, var:bool=True, rot:str|None=None):
    synthesis = BlockManager()
    q = s = tensor(model.encode(query)[0])
    if mlp: q = s = mlp(s)
    i = s.norm().item()

    if max_it is None: max_it = q.shape[0]
    for _ in range(max_it):
        if i < t: break # Break condition: Embedding norm below the norm threshold

        nn = db.read(s, limit=1)[0]
        if nn is None: break  # Break condition: No neighbors

        n = nn.embedding
        s, i, norm_proj = pivot_rotation(q, n, s, i, k, l, method=rot)
        if norm_proj < t and not rot=='rnd': break  # Break condition: Perpendicular embeddings (pivot rotation only)
        synthesis.append_doc(nn)

    synthesis.rearrange(io_order(ast_io_split(synthesis)))
    return merge_variables(model, synthesis) if var else synthesis


def reconstruct(information: float, projections:Tensor, features: int, p:float=0.95):
    docs = projections.size(0)
    max_r, best_mask = 0.0, None
    for l in range(1, min(features, docs) + 1):
        print(l)
        r, mask = best_combination(docs, l, information, projections)
        if r is None: continue
        if r > max_r:
            max_r = r.item()
            best_mask = mask
        if max_r >= p * information: break
    return best_mask.tolist() if best_mask is not None else None


def exhaustive_synthesis(model: LLM, db: BlockDB, query: str, mlp:MLP|None=None):
    q = tensor(model.encode(query)[0])
    if mlp: q = mlp(q)
    i = q.norm().item()
    features = q.shape[0]

    projections = [projection(q, db.get_doc(idx).embedding).norm().item() for idx in range(db.num_docs())]
    projections = tensor(projections)
    r_mask = reconstruct(i, projections, features)
    synthesis = BlockManager()
    for idx in r_mask:
        synthesis.append_doc(db.get_doc(idx))
    synthesis.rearrange(io_order(ast_io_split(synthesis)))

    return synthesis

