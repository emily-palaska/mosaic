from docarray import BaseDoc
from docarray import DocList
from docarray.typing import NdArray
from vectordb import InMemoryExactNNVectorDB, HNSWVectorDB

FEATURE_SIZE = 384

class BlockMergerDoc(BaseDoc):
    label: str = ''
    block: list = []
    embedding: NdArray[FEATURE_SIZE]

class VectorDB:
    def __init__(self, dbtype=HNSWVectorDB, workspace='./'):
        self.db = dbtype[BlockMergerDoc](workspace=workspace)

    def create(self, labels, embeddings, blocks):
        num_values = len(labels)
        doc_list = [BlockMergerDoc(label=labels[i], block=blocks[i], embedding=embeddings[i]) for i in range(num_values)]
        self.db.index(inputs=DocList[BlockMergerDoc](doc_list))

    def read(self, embedding, limit=10):
        if not embedding.ndim == 1:
            embedding = embedding.flatten()

        query = BlockMergerDoc(label='query', block=[], embedding=embedding)

        results = self.db.search(inputs=DocList[BlockMergerDoc]([query]), limit=limit)
        return results[0].matches