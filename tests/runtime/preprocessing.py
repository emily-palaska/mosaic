import os
os.chdir("../../")

from time import time
from datetime import datetime
from llm_blockmerger.load import init_managers, flatten_labels, create_blockdata, nb_variables
from llm_blockmerger.store import BlockDB
from llm_blockmerger.core import plot_sim, norm_cos_sim, LLM

def preprocessing(paths: list, plot=False, db=False):
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


if __name__ == '__main__':
    demo_paths = ['notebooks/example_more.ipynb', 'notebooks/pygrank_snippets.ipynb']
    # paths = ['notebooks/02.02-The-Basics-Of-NumPy-Arrays.ipynb']
    results = 'results/preprocessing.txt'
    timestamp = datetime.now().strftime("%d/%m %H:%M")
    with open(results, 'w') as file:
        file.write(f'Paths: {demo_paths}\nExperiment: {timestamp}\n')

    A = 20
    for i in range(A):
        print(f'\rProgress: {100 * i / A : .2f}%', end='')
        start = time()
        preprocessing(demo_paths)
        with open(results, 'a') as file:
            file.write(f'{time() - start: .3f},')
    print(f'\rExperiment completed')