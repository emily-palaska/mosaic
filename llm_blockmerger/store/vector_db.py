from docarray import DocList
from vectordb import InMemoryExactNNVectorDB, HNSWVectorDB
from llm_blockmerger.core.utils import find_db_files
from docarray import BaseDoc
from docarray.typing import NdArray
from torch.utils.data import Dataset
import torch

def make_doc(feature_size=384):
    class BlockMergerDoc(BaseDoc):
        label: str = ''
        block: list = []
        source: str = ''
        variable_dictionary: dict = {}
        embedding: NdArray[feature_size]

    return BlockMergerDoc

class VectorDB(Dataset):
    def __init__(self,
                 dbtype=HNSWVectorDB,
                 workspace='./databases/',
                 feature_size=384,
                 empty=False):
        assert dbtype in [HNSWVectorDB, InMemoryExactNNVectorDB], "Invalid dbtype"
        self.feature_size = feature_size
        self.BlockMergerDoc = make_doc(feature_size)
        self.db = dbtype[self.BlockMergerDoc](workspace=workspace)
        if empty: empty_docs(workspace=workspace)

    def create(self, labels, blocks, variable_dictionaries, sources, embeddings):
        num_values = len(labels)
        doc_list = [self.BlockMergerDoc(label=labels[i],
                                        block=blocks[i],
                                        source=sources[i],
                                        variable_dictionary=variable_dictionaries[i],
                                        embedding=embeddings[i]) for i in range(num_values)]
        self.db.index(inputs=DocList[self.BlockMergerDoc](doc_list))

    def read(self, embedding, limit=10):
        if not embedding.ndim == 1:
            embedding = embedding.flatten()

        query = self.BlockMergerDoc(label='query', block=[], embedding=embedding)

        results = self.db.search(inputs=DocList[self.BlockMergerDoc]([query]), limit=limit)
        return results[0].matches

    def get_feature_size(self):
        return self.feature_size

    def __len__(self):
        return self.db.num_docs()['num_docs']

    def __getitem__(self, embedding):
        m = self.read(embedding, limit=3)
        return torch.tensor(m[0]), torch.tensor(m[1]), torch.tensor(m[2])

def empty_docs(workspace='./databases/'):
    import sqlite3

    db_files = find_db_files(workspace)

    for db_file in db_files:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM docs")
        conn.commit()
        conn.close()

