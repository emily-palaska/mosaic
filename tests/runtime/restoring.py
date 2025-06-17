import os
os.chdir("../../")

from time import time
from datetime import datetime
from llm_blockmerger.core import LLM
from llm_blockmerger.store import BlockDB

def restoring():
    model = LLM(task='embedding')
    db = BlockDB(empty=False)
    assert db.num_docs() == 31, f'Incorrect number of docs: {db.num_docs()}'

if __name__ == '__main__':
    demo_paths = ['notebooks/example_more.ipynb', 'notebooks/pygrank_snippets.ipynb']
    # paths = ['notebooks/02.02-The-Basics-Of-NumPy-Arrays.ipynb']
    results = 'results/restoring.txt'
    timestamp = datetime.now().strftime("%d/%m %H:%M")
    with open(results, 'w') as file:
        file.write(f'Paths: {demo_paths}\nExperiment: {timestamp}\n[')

    A = 1000
    for i in range(A):
        print(f'\rProgress: {100 * i / A : .2f}%', end='')
        start = time()
        restoring()
        with open(results, 'a') as file:
            file.write(f'{time() - start:.3f}, ')
            if i == A - 1: file.write(']')
    print(f'\rExperiment completed')