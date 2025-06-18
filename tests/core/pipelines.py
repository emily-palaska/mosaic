import os
from torch import stack

from llm_blockmerger.core import LLM, plot_sim, norm_cos_sim, print_synthesis, encoded_json
from llm_blockmerger.load import init_managers, nb_variables, flatten_labels, create_blockdata
from llm_blockmerger.merge import string_synthesis, embedding_synthesis
from llm_blockmerger.store import BlockDB
from tests.core.utils import synthesis_dumb


def preprocess(paths: list, plot=False, db=False):
    managers = init_managers(paths)

    llama = LLM(task='question')
    for manager in managers:
        nb_variables(manager, llama)

    model = LLM(task='embedding')
    embeddings = model.encode(flatten_labels(managers, code=True))
    if plot: plot_sim(norm_cos_sim(embeddings), './plots/similarity_matrix.png')

    if db:
        db = BlockDB(empty=True)
        db.create(embeddings=embeddings, blockdata=create_blockdata(managers, embeddings))
        assert db.num_docs() == len(embeddings), f'{db.num_docs()} != {len(embeddings)}'


def restore():
    model = LLM(task='embedding')
    db = BlockDB(empty=False)
    return model, db


def merge(queries: list, path='./results/synthesis/', save=True, verbose=True):
    model = LLM(task='embedding')
    db = BlockDB(empty=False)
    if verbose: print(f'Initialized BlockDB with {db.num_docs()} docs')

    synthesis = []
    for i, query in enumerate(queries):
        synthesis.append((string_synthesis(model, db, query), embedding_synthesis(model, db, query)))

    for i, s in enumerate(synthesis):
        ss, se = s
        if save:
            qpath = os.path.join(path, f'query{i}')
            synthesis_dumb(ss, queries[i], 'String', qpath + 's.md')
            synthesis_dumb(se, queries[i], 'Embedding', qpath + 'e.md')
        else:
            print_synthesis(ss, queries[i], title='STRING')
            print_synthesis(se, queries[i], title='EMBEDDING')
    if save and verbose: print(f'Results saved in {path}')

def deploy_mlp(model, embeddings, blockdata):
    new_embeddings = stack([model(embedding.to(model.device)) for embedding in embeddings]).tolist()
    new_blockdata = []
    for datadict, new_emb in zip(blockdata, new_embeddings):
        new_datadict = encoded_json(datadict)
        new_datadict['embedding'] = new_emb
        new_blockdata.append(new_datadict)
    db = BlockDB(features=len(new_embeddings[0]), empty=True)
    db.create(embeddings=new_embeddings, blockdata=new_blockdata)


