import os
os.chdir('../')

from datetime import datetime
from time import time

from mosaic.store import BlockDB, ApproxNN, ExactNN
from tests.core.pipelines import restore

def runtime(A=1_000):
    results = 'results/retrieving_run.txt'
    example = 'example query'
    timestamp = datetime.now().strftime("%d/%m %H:%M")
    with open(results, 'w') as file:
        file.write(f'Experiment: {timestamp} A={A}\n')

    for TypeNN in [ApproxNN, ExactNN]:
        model, db = restore(TypeNN)
        assert db.num_docs() > 0, 'Empty BlockDB'
        with open(results, 'a') as file: file.write(f'{TypeNN.__name__.lower()} = [')
        for i in range(A):
            print(f'\r{TypeNN.__name__} Progress: {100 * i / A :.2f}%', end='')
            start = time()
            result = db.read(model.encode(example), limit=1)[0]
            assert result.embedding.shape == (db.features,)
            with open(results, 'a') as file:
                file.write(f'{time() - start:.6f}, ')
                if i == A - 1: file.write(']\n')

    print(f'\rExperiment completed')

def integration(verbose=True):
    for TypeNN in [ApproxNN, ExactNN]:
        db = BlockDB(dbtype=TypeNN, empty=False)
        assert db.num_docs() > 0, 'Empty BlockDB'
        if verbose: print(f'Loaded {TypeNN.__name__} BlockDB with {db.num_docs()} documents')

        example = [0 for _ in range(db.features)]
        result = db.read(example, limit=1)[0]
        assert result.embedding.shape == (db.features,)
        if verbose: print(f'Read example with {result.embedding.shape} features')

if __name__ == '__main__':
    runtime()