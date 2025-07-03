import os
os.chdir("../")

from tests.core.pipelines import preprocess
from time import time
from datetime import datetime

def runtime(A:int=20):
    demo_paths = ['notebooks/example_more.ipynb', 'notebooks/pygrank_snippets.ipynb']
    # paths = ['notebooks/02.02-The-Basics-Of-NumPy-Arrays.ipynb']
    results ='tests/data/preprocessing_run.py'

    with open(results, 'w') as file:
        file.write(f'preprocessing_run = [')

    for i in range(A):
        print(f'\rProgress: {100 * i / A :.2f}%', end='')
        start = time()
        preprocess(demo_paths)
        with open(results, 'a') as file:
            file.write(f'{time() - start:.3f}, ')
            if i == A - 1: file.write(']\n')
    print(f'\rExperiment completed')


def scalability(A:int=20, limits:list|None=None):
    if limits is None: limits = list(range(1, 35, 2))
    demo_paths = ['notebooks/example_more.ipynb', 'notebooks/pygrank_snippets.ipynb']
    # paths = ['notebooks/02.02-The-Basics-Of-NumPy-Arrays.ipynb']
    results ='tests/data/preprocessing_sc.py'
    with open(results, 'w') as file:
        file.write('preprocessing_sc = [')

    for limit in limits:
        with open(results, 'a') as file:
            file.write(f'Limit: {limit}\n[')
        for i in range(A):
            print(f'\rLimit {limit} progress: {100 * i / A :.2f}%', end='')
            start = time()
            preprocess(demo_paths, limit=limit)
            with open(results, 'a') as file:
                file.write(f'{time() - start:.3f}, ')
                if i == A - 1: file.write(']\n')
    print(f'\rExperiment completed')


def integration():
    #demo_paths = ['notebooks/example_more.ipynb', 'notebooks/pygrank_snippets.ipynb']
    #paths = ['notebooks/02.02-The-Basics-Of-NumPy-Arrays.ipynb']
    from tests.data.filenames import more_datafiles
    for data in more_datafiles:
        db = preprocess(data, plot=True, db=True, empty=False)
        print(f'Created database with {db.num_docs()} documents')

if __name__ == '__main__':
    integration()

