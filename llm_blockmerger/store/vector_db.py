from docarray import DocList
from vectordb import InMemoryExactNNVectorDB, HNSWVectorDB
from llm_blockmerger.core.utils import find_db_files, generate_triplets
from docarray import BaseDoc
from docarray.typing import NdArray
from torch.utils.data import Dataset
import torch
import json

def make_doc(feature_size=384):
    class BlockMergerDoc(BaseDoc):
        id: str
        blockdata: str = ''
        embedding: NdArray[feature_size]

    return BlockMergerDoc

class VectorDB(Dataset):
    def __init__(self,
                 databasetype=HNSWVectorDB,
                 workspace='./databases/',
                 feature_size=384,
                 dtype=torch.float32,
                 empty=False):
        assert databasetype in [HNSWVectorDB, InMemoryExactNNVectorDB], "Invalid dbtype"
        self.feature_size = feature_size
        self.dtype = dtype
        self.BlockMergerDoc = make_doc(feature_size)

        if empty: empty_docs(workspace=workspace)
        self.db = databasetype[self.BlockMergerDoc](workspace=workspace, index=True)
        self.triplets = generate_triplets(self.get_num_docs())

    def create(self, embeddings, blockdata):
        num_values = len(embeddings)
        doc_list = [
            self.BlockMergerDoc(
                id=str(self.get_num_docs() +i),
                embedding=embeddings[i],
                blockdata=json.dumps(blockdata[i]),
            )
            for i in range(num_values)
        ]

        self.db.index(inputs=DocList[self.BlockMergerDoc](doc_list))
        self.triplets = generate_triplets(self.get_num_docs())

    def read(self, embedding, limit=10):
        if len(self) == 0:
            raise IndexError("VectorDB is empty")

        if isinstance(embedding, list):
            import numpy as np
            embedding = np.array(embedding)
        if not embedding.ndim == 1:
            embedding = embedding.flatten()

        query = self.BlockMergerDoc(id='', embedding=embedding)

        results = self.db.search(inputs=DocList[self.BlockMergerDoc]([query]), limit=limit)
        return results[0].matches

    def get_feature_size(self):
        return self.feature_size

    def get_num_docs(self):
        return self.db.num_docs()['num_docs']

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        if index >= len(self.triplets):
            raise IndexError(f'Index {index} out of range')

        anchor_idx, positive_idx, negative_idx = self.triplets[index]
        print(anchor_idx, positive_idx, negative_idx)
        anchor = self.db.get_by_id(str(anchor_idx))
        positive = self.db.get_by_id(str(positive_idx))
        negative = self.db.get_by_id(str(negative_idx))

        anchor_embedding = torch.tensor(anchor.embedding, dtype=self.dtype)
        positive_embedding = torch.tensor(positive.embedding, dtype=self.dtype)
        negative_embedding = torch.tensor(negative.embedding, dtype=self.dtype)

        return anchor_embedding, positive_embedding, negative_embedding

def empty_docs(workspace='./databases/'):
    import sqlite3

    db_files = find_db_files(workspace)

    for db_file in db_files:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table[0]  # extract the string from the tuple
            cursor.execute(f"DROP TABLE IF EXISTS `{table_name}`;")  # safer with backticks
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
    sources = ['' for _ in range(len(blocks))]
    variable_dictionaries = [{} for _ in range(len(blocks))]
    data = [{
        'label': labels[i],
        'blocks': blocks[i],
        'variable_dictionary': variable_dictionaries[i],
        'source': sources[i],
    } for i in range(len(blocks))
    ]

    feature_size = 10
    embeddings = [[i for _ in range(feature_size)] for i in range(len(blocks))]

    vector_db = VectorDB(databasetype=HNSWVectorDB,
                         workspace='../../databases/',
                         feature_size=feature_size,
                         empty=False)
    print('Initialized vector database...')
    vector_db.create(embeddings, data)
    print(f'Database entries are {vector_db.get_num_docs()}')
    print(f'Dataset length is {len(vector_db)}')
    result = vector_db.db.get_by_id(str(0))
    print(result.id)
    print(result.embedding)
    blockdata = json.loads(result.blockdata)
    print(blockdata['blocks'])

    print('='*60)




if __name__ == "__main__":
    main()