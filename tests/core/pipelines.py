import os

from llm_blockmerger.core import LLM, plot_sim, norm_cos_sim
from llm_blockmerger.load import init_managers, nb_variables, flatten_labels, create_blockdata
from llm_blockmerger.merge import string_synthesis, embedding_synthesis
from llm_blockmerger.store import BlockDB
from tests.core.utlis import synthesis_dumb


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


def merge(queries: list, path='./results/synthesis/'):
    model = LLM(task='embedding')
    db = BlockDB(empty=False)
    print(f'Initialized BlockDB with {db.num_docs()} docs')
    for i, query in enumerate(queries):
        qpath = os.path.join(path, f'query{i}')
        synthesis = string_synthesis(model, db, query)
        synthesis_dumb(synthesis, query, 'String', qpath + 's.md')
        synthesis = embedding_synthesis(model, db, query)
        synthesis_dumb(synthesis, query, 'Embedding', qpath + 'e.md')
    print(f'Results saved in {path}')

