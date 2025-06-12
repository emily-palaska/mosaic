import os
import numpy as np
from llm_blockmerger.store.blockdb import BlockStore, HNSWVectorDB
os.chdir("../")

def noise_db_input(feature_size=200, instances=10):
    embeddings = np.random.rand(instances, feature_size)
    labels = ['' for _ in range(instances)]
    blocks = [[] for _ in range(instances)]
    return labels, blocks, embeddings

def main():
    feature_size, instances = 150, 10
    labels, blocks, embeddings = noise_db_input(feature_size=feature_size, instances=instances)

    # Create a BlockMergerVectorDB, should initialize empty
    vector_db = BlockStore(dbtype=HNSWVectorDB, feature_size=feature_size, empty=True)
    assert len(vector_db) == 0, 'Initialized BlockMergerVectorDB size is not 0'

    # Try an example and verify the dimensions
    example = np.zeros(feature_size)
    vector_db.create(labels, embeddings, blocks)
    result = vector_db.read(example, limit=1)
    assert result[0].embedding.shape == (feature_size,), "Result should match the embedding's feature shape"
    assert len(vector_db) == instances, 'BlockMergerVectorDB size does not match instances created'

if __name__ == '__main__':
    main()