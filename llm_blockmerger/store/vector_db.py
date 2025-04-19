from docarray import DocList
from vectordb import InMemoryExactNNVectorDB, HNSWVectorDB
from llm_blockmerger.core.utils import find_db_files
from docarray import BaseDoc
from docarray.typing import NdArray
from torch.utils.data import Dataset
import torch

def make_doc(feature_size=384):
    class BlockMergerDoc(BaseDoc):
        id: str
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
        doc_list = [
            self.BlockMergerDoc(
                id=str(i),
                label=labels[i],
                block=blocks[i],
                source=sources[i],
                variable_dictionary=variable_dictionaries[i],
                embedding=embeddings[i]
            )
            for i in range(num_values)
        ]
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

    def __getitem__(self, index):
        # todo make this triplets
        if index >= len(self): raise IndexError(f'Index {index} out of range')
        return self.db.get_by_id(str(index))

def empty_docs(workspace='./databases/'):
    import sqlite3

    db_files = find_db_files(workspace)

    for db_file in db_files:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM docs")
        conn.commit()
        conn.close()

def main():
    blocks = [
        ['    return a + b '],
        ['def add(a, b):\n', '    return a + b '],
        ['print(result)  '],
        ['result = add(2, 3)\n', 'print(result)  '],
        ['print(result) '],
        ['result = add(-1, 1)\n', 'print(result) ']
    ]

    labels = [
        'MARKDOWN: # Functions in python\nSimple addition function\nCOMMENT: This is a simple addition function',
        'MARKDOWN: # Functions in python\nSimple addition function\nCOMMENT: Example of usage:',
        'MARKDOWN: Testing it with two examples\nCOMMENT: Should print 5',
        'MARKDOWN: Testing it with two examples\nCOMMENT: ',
        'MARKDOWN: Testing it with two examples\nCOMMENT: Should print 0',
        'MARKDOWN: Testing it with two examples\nCOMMENT: '
    ]

    variable_dictionaries = [{} for _ in range(len(blocks))]
    feature_size = 10
    sources = ['' for _ in range(len(blocks))]
    embeddings = [[i for _ in range(feature_size)] for i in range(len(blocks))]

    vector_db = VectorDB(dbtype=HNSWVectorDB,
                         workspace='../../databases/',
                         feature_size=feature_size,
                         empty=True)
    print('Initialized vector database...')
    vector_db.create(labels, blocks, variable_dictionaries, sources, embeddings)
    print('Added elements to vector database...')
    print(vector_db[0])

if __name__ == "__main__":
    main()
