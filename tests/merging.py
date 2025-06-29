import os
os.chdir("../")

from time import time
from json import dumps
from numpy import mean, std
from tests.core import merge, restore
from llm_blockmerger.core import LLM, norm_cos_sim
from llm_blockmerger.merge import embedding_synthesis

queries = [
    'Create classifiers with names Regression, SVM, Tree, AdaBoost and Bayes classifiers. Compare them and plot them.',
    #'Initialize a logistic regression model. Use standardization on training inputs. Train the model.',
    'Create a regression model.',
    'Graph operations',
    'How to perform cross_decomposition',
    'Simple PCA algorithm.',
    'Run a PCA algorithm. Visualize it by plotting some plt plots.'
]


def runtime(A=1000):
    results ='results/merging_run.txt'
    with open(results, 'w') as file:
        file.write(f'synthesis = [')
        model, db = restore()
        print(f'Initialized BlockDB with {db.num_docs()} blocks')
        for i in range(A):
            print(f'\rSynthesis Progress: {100 * i / A:.2f}%', end='')
            for query in queries:
                start = time()
                embedding_synthesis(model, db, query)
                file.write(f'{time() - start :.5f}, ')
        file.write(']\nllama = [')
        model = LLM(task='question')
        for i in range(A):
            print(f'\rLlama Progress: {100 * i / A:.2f}%', end='')
            for query in queries:
                start = time()
                model.answer(query)
                file.write(f'{time() - start :.5f}, ')
        file.write(']')
        print('\rExperiment completed')


def gather_codes(synthesis):
    codes = dict()
    for s in synthesis:
        for k, v in s.items():
            try: codes[k].append('\n'.join(v.labels) + '\n' + '\n'.join(v.blocks))
            except KeyError: codes[k] = ['\n'.join(v.labels) + '\n' + '\n'.join(v.blocks)]
    return codes


def quantitative():
    model = LLM(task='embedding')
    synthesis = merge(queries)
    codes = gather_codes(synthesis)
    print(dumps(codes, indent=2))

    qs = [model.encode(query) for query in queries]
    cs = {k: model.encode(b) for k, b in codes.items()}
    sims = {k: norm_cos_sim(qs[i], cs[k]).tolist() for i, k in enumerate(cs.keys())}
    results ='results/merging_quan.txt'
    with open(results, 'w') as file:
        file.write(f'sims = {dumps(sims, indent=2)}')

    for k, v in sims.items(): print(f'{k}: {mean(v)} ± {std(v)}')


def integration():
    merge([queries[0]], save=False)


def qualitive():
    merge(queries)


if __name__ == '__main__':
    quantitative()