import os
os.chdir('../')

from llm_blockmerger.store import BlockDB, ApproxNN, ExactNN
ApproxNN.__name__, ExactNN.__name__ = 'ApproxNN', 'ExactNN'
from llm_blockmerger.core import encoded_json

def runtime(A=20):
    # run each neighbor type for A times and save times
    return NotImplemented

def integration(verbose=True):
    for TypeNN in [ApproxNN, ExactNN]:
        db = BlockDB(dbtype=TypeNN, empty=False)
        assert db.num_docs() > 0, 'Empty BlockDB'
        if verbose: print(f'Loaded {TypeNN.__name__} BlockDB with {db.num_docs()} documents')

        example = [0 for _ in range(db.features)]
        result = db.read(example, limit=1)[0]
        assert result.embedding.shape == (db.features,)
        if verbose: print(f'Read example with {result.embedding.shape} features')


integration()