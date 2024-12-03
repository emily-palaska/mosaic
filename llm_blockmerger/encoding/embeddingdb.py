from docarray import BaseDoc
from docarray import DocList
from docarray.typing import NdArray
from vectordb import InMemoryExactNNVectorDB
from encoding.embeddingmodel import EmbeddingModel

class BlockMergerDoc(BaseDoc):
  text: str = ''
  embedding: NdArray[384]

class BlockMergerEmbeddingDB:
    def __init__(self, workspace='./', model='sentence-transformers/all-MiniLM-L6-v2'):
        # Specify workspace path
        self.db = InMemoryExactNNVectorDB[BlockMergerDoc](workspace=workspace)

        # Create embedding model
        self.embedding_model = EmbeddingModel(model)

    def create(self, labels):
        # Encode labels to extract embeddings
        embeddings = self.embedding_model.encode(labels)

        # Index a list of documents with given labels and embeddings
        num_values = len(labels)
        doc_list = [BlockMergerDoc(text=labels[i], embedding=embeddings[i]) for i in range(num_values)]
        self.db.index(inputs=DocList[BlockMergerDoc](doc_list))

    def read(self, text, limit=10):
        embedding = self.embedding_model.encode(text)
        query = BlockMergerDoc(text='query', embedding=embedding)
        results = self.db.search(inputs=DocList[BlockMergerDoc]([query]), limit=limit)
        return results[0].matches
