from llm_blockmerger.core import remove_common_words, encoded_json, projection, LLM, best_combination, pivot_rotation
from llm_blockmerger.load import BlockManager
from llm_blockmerger.store import BlockDB
from llm_blockmerger.learn import MLP

from llm_blockmerger.merge.merger import merge_variables
from llm_blockmerger.merge.order import synthesis_order

from torch import tensor, Tensor, rand


def check_repetitions(max_rep:int, ids:list, top_nn:list):
    for nn in top_nn:
        if ids.count(nn.id) < max_rep: return nn
    return None


def string_synthesis(model:LLM, db:BlockDB, query:str, max_rep=1, repl:str='[UNK]',
                     var:bool=True, max_it:int|None=None, mlp:MLP|None=None):
    synthesis, ids, last_nn = BlockManager(), [], None
    s = mlp(tensor(model.encode(query))[0]) if mlp else tensor(model.encode(query))[0]

    for _ in range(max_it if max_it else s.shape[0]):
        nn = check_repetitions(max_rep, ids, db.read(s, limit=3))
        if nn is None or nn == last_nn: break  # Break condition: No valid neighbors

        last_nn = nn
        ids.append(nn.id)

        nn_label = encoded_json(nn.blockdata)['label']
        new_query = remove_common_words(og=query, rem=nn_label, repl=repl)
        if new_query == query: break # Break condition: Unchanged specification

        synthesis.append_doc(nn)
        query = new_query
        s = mlp(tensor(model.encode(query))) if mlp else tensor(model.encode(query))

    synthesis.rearrange(synthesis_order(synthesis))
    return merge_variables(model, synthesis) if var else synthesis


def embedding_synthesis(model:LLM, db:BlockDB, query:str, k:float=0.9, l:float=1.4, t:float=0.05,
                        max_it:int|None=None, mlp:MLP|None=None, var:bool=True, rot:str|None=None):
    synthesis, last_nn = BlockManager(), None
    q = mlp(tensor(model.encode(query))[0]) if mlp else tensor(model.encode(query))[0]
    s = rand(q.shape) if rot == 'rnd' else q
    i = q.norm().item()

    for _ in range(max_it if max_it else q.shape[0]):
        if i < t: break # Break condition: Information norm below the norm threshold

        nn = db.read(s, limit=1)[0]
        if nn is None or nn == last_nn: break  # Break condition: No neighbors

        n = nn.embedding
        s, i, norm_proj = pivot_rotation(q, n, s, i, k, l, method=rot)
        if norm_proj < t and not rot=='rnd': break  # Break condition: Perpendicular embeddings (not for rnd)
        synthesis.append_doc(nn)
        last_nn = nn

    synthesis.rearrange(synthesis_order(synthesis))
    return merge_variables(model, synthesis) if var else synthesis


def reconstruct(information: float, projections:Tensor, features: int, p:float=0.95, max_l=3):
    docs = projections.size(0)
    best_r, best_mask = 0.0, None
    iterations = min(min(features, docs), max_l) + 1
    for l in range(1, iterations):
        r, mask = best_combination(docs, l, information, projections)
        if r is None: continue
        if r > best_r:
            best_r, best_mask  = r, mask
        if best_r >= p * information: break
    return best_mask.tolist() if best_mask is not None else None


def exhaustive_synthesis(model: LLM, db: BlockDB, query: str, mlp:MLP|None=None, noise:bool=False):
    if noise: return BlockManager()
    q = mlp(tensor(model.encode(query))[0]) if mlp else tensor(model.encode(query))[0]
    i = q.norm().item()

    ns = [db.get_doc(idx).embedding for idx in range(db.num_docs())]
    projs = tensor([projection(q, n).norm().item() for n in ns])

    synthesis = BlockManager()
    for idx in reconstruct(i, projs, q.shape[0]): synthesis.append_doc(db.get_doc(idx))
    synthesis.rearrange(synthesis_order(synthesis))

    return synthesis

