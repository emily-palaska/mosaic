import os
os.chdir("../")

from tests.core.pipelines import preprocess
from time import time
from datetime import datetime


def runtime(A=20):
    demo_paths = ['notebooks/example_more.ipynb', 'notebooks/pygrank_snippets.ipynb']
    # paths = ['notebooks/02.02-The-Basics-Of-NumPy-Arrays.ipynb']
    results = 'results/preprocessing.txt'
    timestamp = datetime.now().strftime("%d/%m %H:%M")
    with open(results, 'w') as file:
        file.write(f'Paths: {demo_paths}\nExperiment: {timestamp}\n')

    for i in range(A):
        print(f'\rProgress: {100 * i / A : .2f}%', end='')
        start = time()
        preprocess(demo_paths)
        with open(results, 'a') as file:
            file.write(f'{time() - start: .3f},')
    print(f'\rExperiment completed')


def integration():
    demo_paths = ['notebooks/example_more.ipynb', 'notebooks/pygrank_snippets.ipynb']
    # paths = ['notebooks/02.02-The-Basics-Of-NumPy-Arrays.ipynb']
    preprocess(demo_paths, plot=True, db=True)

