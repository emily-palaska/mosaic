import os
os.chdir("../")
from llm_blockmerger.linear_merging import *

def main():
    # Create noise as embedding inputs
    feature_size = 200
    instances_size = 10
    embeddings = np.random.rand(instances_size, feature_size)
    labels = ['' for _ in range(instances_size)]
    blocks = [[] for _ in range(instances_size)]

    import llm_blockmerger.encoding.config
    llm_blockmerger.encoding.config.FEATURE_SIZE = feature_size

    from llm_blockmerger.encoding.vector_db import VectorDB, HNSWVectorDB

    # Create a vector database, and try it with an add/read example
    vector_db = (VectorDB(dbtype=HNSWVectorDB))
    print(vector_db.get_size())

    example = np.zeros(feature_size)
    vector_db.create(labels, embeddings, blocks)
    vector_db.read(example)
    print(vector_db.get_size())



if __name__ == '__main__':
    main()