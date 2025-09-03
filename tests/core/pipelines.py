from torch import stack, Tensor
from os.path import join

from mosaic.core import LLM, plot_sim, norm_cos_sim, print_synthesis, encoded_json
from mosaic.load import init_managers, nb_variables, flatten_labels, create_blockdata
from mosaic.merge import string_synthesis, embedding_synthesis, exhaustive_synthesis, baseline_synthesis
from mosaic.store import BlockDB, ApproxNN, ExactNN
from mosaic.learn import MLP
from tests.core.utils import md_dumb_synthesis, slice_2d


def preprocess(paths: list, plot:bool=False, db:bool=False, limit:int|None=None, empty:bool=True):
    managers = init_managers(paths)
    if limit: managers = slice_2d(managers, limit)

    llama = LLM(task='question')
    for manager in managers:
        nb_variables(manager, llama)
        print(manager)

    model = LLM(task='embedding')
    embeddings = model.encode(flatten_labels(managers, code=True))
    if plot: plot_sim(norm_cos_sim(embeddings), './plots/similarity_matrix.png')

    if db:
        blockdb = BlockDB(empty=empty)
        docs = blockdb.num_docs()
        blockdb.create(embeddings=embeddings, blockdata=create_blockdata(managers, embeddings))
        assert blockdb.num_docs() == len(embeddings) + docs, f'{blockdb.num_docs()} != {len(embeddings) + docs}'
        return blockdb
    return None


def restore(dbtype:type[ApproxNN|ExactNN]=ApproxNN):
    model = LLM(task='embedding')
    db = BlockDB(dbtype=dbtype, empty=False)
    return model, db


def merge(queries:list, path:str='./results/synthesis/', model:LLM|None=None, db:BlockDB|None=None,
          mlp:MLP|None=None, save:bool=True, verbose:bool=True):
    names = {'s': 'String', 'e': 'Embedding', 're': 'Reverse Embedding', 'rnd': 'Random', 'x': 'Exhaustive', 'b': 'Baseline'}
    if model is None: model = LLM(task='embedding')
    if db is None: db = BlockDB(empty=False)
    if verbose: print(f'Initialized BlockDB with {db.num_docs()} docs')
    llama = LLM(task='question')

    results = []
    for i, query in enumerate(queries):
        if verbose: print(f'\rProgress: {i}/{len(queries)}', end='')
        methods = {
            's': string_synthesis(model, db, query, mlp=mlp),
            'e': embedding_synthesis(model, db, query, mlp=mlp),
            're': embedding_synthesis(model, db, query, mlp=mlp, rot='rev'),
            'rnd': embedding_synthesis(model, db, query, mlp=mlp, rot='rnd'),
            'x': exhaustive_synthesis(model, db, query, mlp=mlp),
            'b': baseline_synthesis(llama, query)
        }
        results.append(methods)

        for key, synthesis in methods.items():
            qpath = join(path, f'{names[key].lower()}/query{i}')
            if save: md_dumb_synthesis(synthesis, query, names[key], qpath + f'{key}.md')
            #elif verbose: print_synthesis(synthesis, query, title=names[key].upper())

    if save and verbose: print(f'\rResults saved in {path}')
    elif verbose: print(f'\rMerging completed.')
    return results


def deploy_mlp(model:MLP, embeddings:Tensor, blockdata:str):
    new_embeddings = stack([model(embedding.to(model.device)) for embedding in embeddings]).tolist()
    blockdata = encoded_json(blockdata)
    new_blockdata = []
    for datadict, new_emb in zip(blockdata, new_embeddings):
        new_datadict = encoded_json(datadict)
        new_datadict['embedding'] = new_emb
        new_blockdata.append(new_datadict)
    db = BlockDB(features=len(new_embeddings[0]), empty=True)
    db.create(embeddings=new_embeddings, blockdata=new_blockdata)


