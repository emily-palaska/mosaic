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
    'Initialize a logistic regression model. Use standardization on training inputs. Train the model.',
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


def separate_methods(synthesis_list):
    results = dict()
    for synthesis_dict in synthesis_list:
        for m, s in synthesis_dict.items():
            try: results[m].append('\n'.join(s.labels) + '\n' + '\n'.join(s.blocks))
            except KeyError: results[m] = ['\n'.join(s.labels) + '\n' + '\n'.join(s.blocks)]
    return results


def quantitative():
    model = LLM(task='embedding')
    synthesis_list = merge(queries)
    results = separate_methods(synthesis_list)

    qs = [model.encode(query) for query in queries]
    rs = {k: model.encode(r) for k, r in results.items()}
    sims = {k: norm_cos_sim(qs[i], rs[k]).tolist() for i, k in enumerate(rs.keys())}

    results ='results/merging_quan.txt'
    with open(results, 'w') as file:
        file.write(f'sims = {dumps(sims, indent=2)}')

    print('Average similarities')
    for k, v in sims.items(): print(f'{k}: {mean(v)} ± {std(v)}')
    print('Blocks:')
    for i, synthesis_dict in enumerate(synthesis_list):
        print(f'\tQ{i}: ', end='')
        for m, s in synthesis_dict.items():
            print(f'{m}={len(s)}', end=' ')
        print('')


def integration():
    merge([queries[0]], save=False)


def qualitive():
    merge(queries)


if __name__ == '__main__':
    quantitative()