from docarray import DocList
from vectordb import InMemoryExactNNVectorDB, HNSWVectorDB
from llm_blockmerger.core.utils import find_db_files
from docarray import BaseDoc
from docarray.typing import NdArray

def make_doc(feature_size=384):
    class BlockMergerDoc(BaseDoc):
        label: str = ''
        block: list = []
        source: str = ''
        variables: list = []
        var_descriptions: list = []
        embedding: NdArray[feature_size]

    return BlockMergerDoc

class VectorDB:
    def __init__(self,
                 dbtype=HNSWVectorDB,
                 workspace='./databases/',
                 feature_size=384,
                 empty=False):
        assert dbtype in [HNSWVectorDB, InMemoryExactNNVectorDB], "Invalid dbtype"
        self.BlockMergerDoc = make_doc(feature_size)
        self.db = dbtype[self.BlockMergerDoc](workspace=workspace)
        if empty: empty_docs(workspace=workspace)

    def create(self, labels, blocks, variables, var_descriptions, sources, embeddings):
        num_values = len(labels)
        doc_list = [self.BlockMergerDoc(label=labels[i],
                                        block=blocks[i],
                                        source=sources[i],
                                        variables=variables[i],
                                        var_descriptions = var_descriptions[i],
                                        embedding=embeddings[i]) for i in range(num_values)]
        self.db.index(inputs=DocList[self.BlockMergerDoc](doc_list))

    def read(self, embedding, limit=10):
        if not embedding.ndim == 1:
            embedding = embedding.flatten()

        query = self.BlockMergerDoc(label='query', block=[], embedding=embedding)

        results = self.db.search(inputs=DocList[self.BlockMergerDoc]([query]), limit=limit)
        return results[0].matches

    def get_size(self):
        return self.db.num_docs()['num_docs']

def empty_docs(workspace='./databases/'):
    import sqlite3

    db_files = find_db_files(workspace)

    for db_file in db_files:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM docs")
        conn.commit()
        conn.close()

