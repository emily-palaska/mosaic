import os
from torch import stack

from llm_blockmerger.core import LLM, plot_sim, norm_cos_sim, print_synthesis, encoded_json
from llm_blockmerger.load import init_managers, nb_variables, flatten_labels, create_blockdata
from llm_blockmerger.merge import string_synthesis, embedding_synthesis, exhaustive_synthesis
from llm_blockmerger.store import BlockDB
from llm_blockmerger.learn import MLP
from tests.core.utils import md_dumb_synthesis


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


def merge(queries: list, path='./results/synthesis/', mlp:MLP|None=None, save=True, verbose=True):
    methods = {'s': 'String', 'e': 'Embedding', 'x': 'Exhaustive'}
    model = LLM(task='embedding')
    db = BlockDB(empty=False)
    if verbose: print(f'Initialized BlockDB with {db.num_docs()} docs')

    query_synthesis = []
    for query in queries:
        query_synthesis.append((
            string_synthesis(model, db, query, mlp=mlp),
            embedding_synthesis(model, db, query, mlp=mlp),
            exhaustive_synthesis(model, db, query, mlp=mlp)
        ))

    for i, synthesis in enumerate(query_synthesis):
        qpath = os.path.join(path, f'query{i}')
        for m, s in zip(methods, synthesis):
            if save:
                md_dumb_synthesis(s, queries[i], methods[m], qpath + f'{m}.md')
            else: print_synthesis(s, queries[i], title=methods[m].upper())
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


