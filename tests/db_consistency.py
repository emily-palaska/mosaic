import os
import numpy as np
from llm_blockmerger.store.blockdb import BlockDB
os.chdir("../")

def noise_db_input(features=200, instances=10):
    embeddings = np.random.rand(instances, features)
    blockdata = [{} for _ in range(instances)]
    return embeddings, blockdata

def main():
    features, instances = 150, 10
    embeddings, blockdata= noise_db_input(features=features, instances=instances)

    db = BlockDB(features=features, empty=True)
    assert len(db) == 0, 'Initialized BlockDB size is not 0'

    example = np.zeros(features)
    db.create(embeddings, blockdata)
    result = db.read(example, limit=1)
    assert result[0].embedding.shape == (features,), "Result should match the embedding's feature shape"
    assert len(db) == instances, 'BlockMergerVectorDB size does not match instances created'

if __name__ == '__main__':
    main()