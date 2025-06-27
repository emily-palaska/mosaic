import os
os.chdir("../")

from tests.core.pipelines import preprocess
from time import time
from datetime import datetime


def runtime(A:int=20):
    demo_paths = ['notebooks/example_more.ipynb', 'notebooks/pygrank_snippets.ipynb']
    # paths = ['notebooks/02.02-The-Basics-Of-NumPy-Arrays.ipynb']
    results ='results/preprocessing_run.txt'
    timestamp = datetime.now().strftime("%d/%m %H:%M")
    with open(results, 'w') as file:
        file.write(f'Paths: {demo_paths}\nExperiment: {timestamp}\n[')

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
    results ='results/preprocessing_sc.txt'
    timestamp = datetime.now().strftime("%d/%m %H:%M")
    with open(results, 'w') as file:
        file.write(f'Paths: {demo_paths}\nExperiment: {timestamp}\n')

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
    # paths = ['notebooks/02.02-The-Basics-Of-NumPy-Arrays.ipynb']
    classification = [
        'notebooks/dataset2/classification/plot_classifier_comparison.ipynb',
        'notebooks/dataset2/classification/plot_classification_probability.ipynb',
        'notebooks/dataset2/classification/plot_digits_classification.ipynb',
        'notebooks/dataset2/classification/plot_lda.ipynb',
        'notebooks/dataset2/classification/plot_lda_qda.ipynb'
    ]
    cross_decomposition = [
        'notebooks/dataset2/cross_decomposition/plot_compare_cross_decomposition.ipynb',
        'notebooks/dataset2/cross_decomposition/plot_pcr_vs_pls.ipynb'
    ]
    db = preprocess(classification, plot=True, db=True, empty=False)
    print(f'Created database with {db.num_docs()} documents')

if __name__ == '__main__':
    integration()

