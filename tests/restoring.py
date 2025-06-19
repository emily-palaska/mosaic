import os
os.chdir("../")

from time import time
from datetime import datetime
from tests.core import restore

def runtime(A=1000, verbose=True):
    demo_paths = ['notebooks/example_more.ipynb', 'notebooks/pygrank_snippets.ipynb']
    # paths = ['notebooks/02.02-The-Basics-Of-NumPy-Arrays.ipynb']
    results = 'results/restoring.txt'
    timestamp = datetime.now().strftime("%d/%m %H:%M")
    with open(results, 'w') as file:
        file.write(f'Paths: {demo_paths}\nExperiment: {timestamp}\n[')

    for i in range(A):
        if verbose: print(f'\rProgress: {100 * i / A : .2f}%', end='')
        start = time()
        model, db = restore()
        assert db.num_docs() == 31, f'Incorrect number of docs: {db.num_docs()}'
        with open(results, 'a') as file:
            file.write(f'{time() - start:.3f}, ')
            if i == A - 1: file.write(']')
    if verbose: print(f'\rExperiment completed')


def integration():
    model, db = restore()
    print(f'Loaded {model.name} model and BlockDB with {db.num_docs()} docs and {db.features} features')

if __name__ == '__main__':
    integration()