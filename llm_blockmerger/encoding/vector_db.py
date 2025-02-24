from docarray import DocList
from vectordb import InMemoryExactNNVectorDB, HNSWVectorDB
from .blockmergerdoc import BlockMergerDoc
import os

class VectorDB:
    def __init__(self, dbtype=HNSWVectorDB, workspace='./databases/'):
        self.db = dbtype[BlockMergerDoc](workspace=workspace)
        empty_docs(workspace=workspace)

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

def find_db_files(folder_path):
    # List to store all .db file paths
    db_files = []

    # Walk through the folder and its subfolders
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.db'):
                # Append the full path of the .db file
                db_files.append(os.path.join(root, file))

    return db_files