from llm_blockmerger.load import BlockManager
from llm_blockmerger.core import remove_common_words, encoded_json, projection, LLM, ast_io_split
from llm_blockmerger.merge.merger import merge_variables
from llm_blockmerger.merge.order import  io_order
from llm_blockmerger.store import BlockDB
from torch import norm, tensor


def string_synthesis(model: LLM, db: BlockDB, spec: str, max_rep=2, max_it=10, repl='[UNK]', var=True):
    synthesis, ids = BlockManager(), []

    for _ in range(max_it):
        if spec in ['', ' ']: return synthesis # Break condition: Empty specification

        s = model.encode(spec)
        nn = check_repetitions(max_rep, ids, db.read(s, limit=3))
        if nn is None: break  # Break condition: No valid neighbors

        ids.append(nn.id)
        nn_label = encoded_json(nn.blockdata)['label']
        new_spec = remove_common_words(og=spec, rem=''.join(nn_label), repl=repl)
        if new_spec == spec: break # Break condition: Unchanged specification

        synthesis.append_doc(nn)
        spec = new_spec

    synthesis.rearrange(io_order(ast_io_split(synthesis)))
    return merge_variables(model, synthesis) if var else synthesis


def embedding_synthesis(model: LLM, db: BlockDB, spec: str, k=0.9, l=1.4, max_it=10, t=0.05, var=True):
    synthesis = BlockManager()
    s = tensor(model.encode(spec)[0])
    spec_emb = tensor(model.encode(spec)[0])
    i = spec_emb.norm().item()

    for _ in range(max_it):
        if i < t: break # Break condition: Embedding norm below the norm threshold

        nn = db.read(s, limit=1)[0]
        if nn is None: break  # Break condition: No neighbors

        n = nn.embedding
        print(n.shape)
        n_proj = projection(n, s)
        i_proj = projection(spec_emb, n)
        if norm(n_proj) < t: break  # Break condition: Perpendicular embeddings

        synthesis.append_doc(nn)

        s = l * n_proj - s
        s /= s.norm()
        i -= k * i_proj.norm().item()

    synthesis.rearrange(io_order(ast_io_split(synthesis)))
    return merge_variables(model, synthesis) if var else synthesis


def check_repetitions(max_rep:int, ids:list, top_nn:list):
    for neighbor in top_nn:
        if ids.count(neighbor.id) < max_rep:
            return neighbor
    return None