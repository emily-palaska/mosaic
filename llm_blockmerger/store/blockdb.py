from docarray import DocList
from vectordb import InMemoryExactNNVectorDB as ExactNN
from vectordb import HNSWVectorDB as ApproxNN
from torch.utils.data import Dataset
from torch import tensor, stack, float32
from json import dumps

from llm_blockmerger.core import triplets
from llm_blockmerger.store.docs import doc_class, get_docs, empty_docs, separate_docs, find_docs

class BlockDB(Dataset):
    def __init__(self,
                 dbtype=ApproxNN,
                 workspace='./databases/',
                 features=384,
                 precision=float32,
                 empty=False):
        assert dbtype in [ApproxNN, ExactNN], "Invalid dbtype"

        self.dbtype = dbtype
        self.ext = '.db' if self.dbtype == ApproxNN else '.bin'
        self.workspace = workspace
        self.features = features
        self.precision = precision
        self.BlockDoc = doc_class(features)

        if empty: self._init_empty()
        elif dbtype==ApproxNN: self._restore_approx()
        else: self._restore_exact()
        self.triplets = triplets(self.num_docs())

    def _init_empty(self):
        empty_docs(self.workspace, self.ext)
        self.db = self.dbtype[self.BlockDoc](
            workspace=self.workspace,
            index=True,
            ef=200
        )
        assert self.num_docs() == 0, f"BlockDB didn't initialize empty, got {self.num_docs()} entries"

    def _restore_approx(self):
        files = find_docs(self.workspace, self.ext)
        assert len(files) == 1, f"Multiple db files found in workspace {self.workspace}: {files}"
        embeddings, blockdata = separate_docs(get_docs(files[0]))
        if (features := len(embeddings[0])) != self.features:
            self.features = features
            self.BlockDoc = doc_class(features)
        self._init_empty()
        self.create(embeddings, blockdata)
        assert self.db.num_docs() != 0, f"BlockDB wasn't restored successfully, got {self.db.num_docs()} entries"

    def _restore_exact(self):
        self.db = self.dbtype[self.BlockDoc](
            workspace=self.workspace,
            index=True
        )
        assert self.db.num_docs() != 0, f"BlockDB wasn't restored successfully, got {self.db.num_docs()} entries"

    def create(self, embeddings, blockdata):
        doc_list = [
            self.BlockDoc(
                id=str(self.num_docs() + i),
                embedding=tensor(embeddings[i], dtype=self.precision),
                blockdata=dumps(blockdata[i]),
            )
            for i in range(len(embeddings))
        ]
        self.db.index(sapce='cosine', inputs=DocList[self.BlockDoc](doc_list))
        self.triplets = triplets(self.num_docs())
        self.db.persist()

    def read(self, embedding, limit=10):
        if self.num_docs() == 0: raise IndexError("BlockDB is empty")

        embedding = tensor(embedding, dtype=self.precision)
        if embedding.ndim > 1:
            embedding = embedding.flatten()

        query = self.BlockDoc(id='', embedding=embedding)
        results = self.db.search(inputs=DocList[self.BlockDoc]([query]), limit=limit)
        return results[0].matches

    def num_docs(self):
        return self.db.num_docs()['num_docs']

    def embeddings(self):
        return stack([self.db.get_by_id(str(i)).embedding for i in range(self.num_docs())])

    def blockdata(self):
        return [self.db.get_by_id(str(i)).blockdata for i in range(self.num_docs())]

    def get_doc(self, i):
        return self.db.get_by_id(str(i))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        if index >= self.__len__(): raise IndexError(f'Index {index} out of range')

        docs = [self.db.get_by_id(str(idx)) for idx in self.triplets[index]]
        return [doc.embedding for doc in docs]