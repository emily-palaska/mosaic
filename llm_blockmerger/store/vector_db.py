from docarray import DocList
from vectordb import InMemoryExactNNVectorDB, HNSWVectorDB
import numpy as np
from torch.utils.data import Dataset
import torch, json

from llm_blockmerger.store.doc_operations import (
    find_db_files,
    make_doc,
    generate_triplets,
    get_db_rows,
    extract_rows_content,
    empty_docs
)

class VectorDB(Dataset):
    def __init__(self,
                 databasetype=HNSWVectorDB,
                 workspace='./databases/',
                 feature_size=384,
                 dataset_dtype=torch.float32,
                 empty=False):
        assert databasetype in [HNSWVectorDB, InMemoryExactNNVectorDB], "Invalid dbtype"

        self.databasetype = databasetype
        self.workspace = workspace
        self.feature_size = feature_size
        self.dataset_dtype = dataset_dtype
        self.BlockMergerDoc = make_doc(feature_size)

        if empty: self._initialize_empty_db()
        else: self._restore_db()
        self.triplets = generate_triplets(self.get_num_docs())

    def _initialize_empty_db(self):
        empty_docs(workspace=self.workspace)
        self.db = self.databasetype[self.BlockMergerDoc](
            workspace=self.workspace,
            index=True,
            ef=200
        )
        assert self.get_num_docs() == 0, f"VectorDB didn't initialize empty, got {self.get_num_docs()} entries"

    def _restore_db(self):
        db_files = find_db_files(self.workspace)
        assert len(db_files) == 1, f"Multiple db files found in workspace {self.workspace}: {db_files}"
        embeddings, blockdata = extract_rows_content(get_db_rows(db_files[0]))
        self._initialize_empty_db()
        self.create(embeddings, blockdata)

    def create(self, embeddings, blockdata):
        doc_list = [
            self.BlockMergerDoc(
                id=str(self.get_num_docs() +i),
                embedding=torch.tensor(embeddings[i], dtype=self.dataset_dtype),
                blockdata=json.dumps(blockdata[i]),
            )
            for i in range(len(embeddings))
        ]
        self.db.index(inputs=DocList[self.BlockMergerDoc](doc_list))
        self.triplets = generate_triplets(self.get_num_docs())
        self.db.persist()

    def read(self, embedding, limit=10):
        if self.get_num_docs() == 0:
            raise IndexError("VectorDB is empty")

        embedding = torch.tensor(embedding, dtype=self.dataset_dtype)
        if embedding.ndim > 1:
            embedding = embedding.flatten()

        query = self.BlockMergerDoc(id='', embedding=embedding)
        results = self.db.search(inputs=DocList[self.BlockMergerDoc]([query]), limit=limit)
        return results[0].matches

    def get_num_docs(self):
        return self.db.num_docs()['num_docs']

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        if index >= len(self.triplets):
            raise IndexError(f'Index {index} out of range')
        indices = self.triplets[index]
        docs = [self.db.get_by_id(str(idx)) for idx in indices]
        embeddings = [doc.embedding for doc in docs]
        return embeddings

def main():
    num_entries = 10
    feature_size = 10
    empty = True

    embeddings = [
        [i for _ in range(feature_size)]
        for i in range(num_entries)
    ]

    data = [{
        'label': f'This is label {i}',
        'blocks': ['This is', f'block {i}'],
        'variable_dictionary': {i:f'This is dictionary {i}'},
        'source': f'This is source {i}',
        'embedding': embeddings[i]
    } for i in range(num_entries)
    ]

    dummy_embedding = np.array([0 for _ in range(feature_size)])

    vector_db = VectorDB(databasetype=HNSWVectorDB,
                         workspace=r"D:\Σχολή\Διπλωματική\LlmBlockMerger-Diploma\databases",
                         feature_size=feature_size,
                         empty=empty)

    if empty: vector_db.create(embeddings, data)

    print(f'Database entries are {vector_db.get_num_docs()}')
    print(f'Dataset length is {len(vector_db)}')

    print('\n' + '=' * 60)
    print(vector_db.read(dummy_embedding))

    print('\n' + '=' * 60)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(vector_db, batch_size=32, shuffle=False)
    for _ in train_loader: pass
    print('Successful train loader pass...')

if __name__ == "__main__":
    main()