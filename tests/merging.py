import os
os.chdir("../")

from time import time
from numpy import mean, std
from random import randint, sample
from json import dumps

from tests.core import merge, restore, separate_methods
from llm_blockmerger.core import LLM, norm_cos_sim
from llm_blockmerger.store import BlockDB
from llm_blockmerger.merge import embedding_synthesis
from tests.data import queries

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


def quantitative(db=None, sample_q=None, filename=None):
    demo_queries = sample(queries, sample_q) if sample_q else queries
    model = LLM(task='embedding')
    results = dict()
    info = separate_methods(merge(demo_queries, save=False, verbose=True, db=db))

    qs = [model.encode(query) for query in demo_queries]
    cs = {method: model.encode(m_dict["codes"]) for method, m_dict in info.items()}
    for method, m_dict in info.items():
        results[method] = {
            "sims": [norm_cos_sim(q, c).item() for q, c in zip(qs, cs[method])],
            "blocks": info[method]["blocks"],
            "lines": info[method]["lines"]
        }

    print('\tAverage similarities')
    for m, m_dict in results.items(): print(f'\t\t{m}: {mean(m_dict["sims"])} ± {std(m_dict["sims"])}')
    print('\tAverage Blocks:')
    for m, m_dict in results.items(): print(f'\t\t{m}: {mean(m_dict["blocks"])} ± {std(m_dict["blocks"])}')
    print('\tAverage Lines:')
    for m, m_dict in results.items(): print(f'\t\t{m}: {mean(m_dict["lines"])} ± {std(m_dict["lines"])}')

    if filename:
        with open(filename, 'w') as file:
            file.write(f'merging_quan = {dumps(results, indent=2)}')
    return results


def scalability(step_size=100):
    filename, results = 'tests/core/graphs/merging_sc.py', dict()

    db = BlockDB(empty=False)
    embeddings, blockdata = db.embeddings(), db.blockdata()
    print(f'Loaded BlockDB with {db.num_docs()} docs')

    for step in range(step_size, db.num_docs(), step_size):
        print(f'Samples: {step}')
        sample_ids = sorted([randint(0, len(embeddings)-1) for _ in range(step)])
        sample_embeddings = embeddings[sample_ids]
        sample_blockdata = [blockdata[s] for s in sample_ids]
        db = BlockDB(empty=True)
        db.create(sample_embeddings, sample_blockdata)
        for method, m_dict in quantitative(db).items():
            if not method in results: results[method] = {"sims": dict(), "blocks": dict()}
            results[method]["sims"][step] = m_dict["sims"]
            results[method]["blocks"][step] = m_dict["blocks"]
    with open(filename, 'w') as file: file.write(f'merging_sc = {dumps(results, indent=2)}')

    db = BlockDB(empty=True)
    db.create(embeddings, blockdata)
    print(f'Restored BlockDB with {db.num_docs()} docs')


def integration():
    merge([queries[286]], save=False, verbose=True)


def qualitive():
    small_dataset_queries = [
        'Initialize a logistic regression model. Use standardization on training inputs. Train the model.',
        'Create a regression model.',
        'Simple Graph operations',
        'Simple PCA algorithm.',
        'How do you normalize data?'
    ]
    merge(small_dataset_queries)


if __name__ == '__main__':
    filename = 'tests/graphs/merging_quan.py'
    qualitive()