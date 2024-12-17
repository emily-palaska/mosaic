from docarray import BaseDoc
from docarray import DocList
from docarray.typing import NdArray
from vectordb import InMemoryExactNNVectorDB

class BlockMergerDoc(BaseDoc):
    """
    A document class representing an entry in the vector database.
    Each document has:
    - `text`: A string field for storing textual data (e.g., labels or descriptions).
    - `embedding`: An NdArray field for storing the corresponding vector embedding.
    """
    label: str = ''
    embedding: NdArray
    block: list

def initialize_vectordb(workspace='./'):
    """
    Initialize an in-memory exact nearest neighbor vector database.

    :param workspace: Path to the workspace directory for the database (default is current directory).
    :return: An instance of InMemoryExactNNVectorDB specific to BlockMergerDoc.
    """
    return InMemoryExactNNVectorDB[BlockMergerDoc](workspace=workspace)

def vectordb_create(db, labels, embeddings, blocks):
    """
    Index a list of documents in the vector database.

    :param db: The initialized vector database instance.
    :param labels: A list of text labels for the documents.
    :param embeddings: A list of embeddings (vectors) corresponding to the labels.
    :param blocks: A list of blocks containing code corresponding to the labels.
    """
    num_values = len(labels)
    doc_list = [BlockMergerDoc(label=labels[i], embedding=embeddings[i], block=blocks[i]) for i in range(num_values)]
    db.index(inputs=DocList[BlockMergerDoc](doc_list))

def vectordb_read(db, embedding, limit=10):
    """
    Perform a nearest neighbor search in the vector database using a query embedding.

    :param db: The initialized vector database instance.
    :param embedding: The query embedding (vector) to search for similar entries.
    :param limit: The maximum number of nearest neighbors to return (default is 10).
    :return: A list of matching documents sorted by similarity.
    """
    # Ensure the embedding is 1D (flatten if necessary)
    if not embedding.ndim == 1:
        embedding = embedding.flatten()

    # Perform the search and return the results
    query = BlockMergerDoc(label='', embedding=embedding, block=[])
    results = db.search(inputs=DocList[BlockMergerDoc]([query]), limit=limit)
    return results[0].matches
