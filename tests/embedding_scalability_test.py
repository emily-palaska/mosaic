import os
os.chdir("../")
from llm_blockmerger.linear_merging import *

def noise_db_input(feature_size=200, instances=10):
    embeddings = np.random.rand(instances, feature_size)
    labels = ['' for _ in range(instances)]
    blocks = [[] for _ in range(instances)]
    return labels, blocks, embeddings

def main():
    feature_size, instances = 150, 10
    labels, blocks, embeddings = noise_db_input(feature_size=feature_size, instances=instances)

    # Dynamically change the size of the BaseDoc before creating the VectorDB
    import llm_blockmerger.encoding.config
    llm_blockmerger.encoding.config.FEATURE_SIZE = feature_size

    # Create a VectorDB, should initialize empty
    from llm_blockmerger.encoding.vector_db import VectorDB, HNSWVectorDB
    vector_db = (VectorDB(dbtype=HNSWVectorDB))
    assert vector_db.get_size() == 0, 'Initialized VectorDB size is not 0'

    # Try an example and verify the dimensions
    example = np.zeros(feature_size)
    vector_db.create(labels, embeddings, blocks)
    result = vector_db.read(example, limit=1)
    assert result[0].embedding.shape == (feature_size,), "Result should match the embedding's feature shape"
    assert vector_db.get_size() == instances, 'VectorDB size does not match instances created'

if __name__ == '__main__':
    main()