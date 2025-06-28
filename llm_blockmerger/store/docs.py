import os
from sqlite3 import connect
from docarray import BaseDoc
from docarray.typing import TorchTensor

from llm_blockmerger.core import encoded_json

def doc_class(features=384):
    class BlockDoc(BaseDoc):
        id: str = ''
        embedding: TorchTensor[features]
        blockdata: str = ''

    return BlockDoc

def get_docs(path:str, table='docs'):
    conn = connect(path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table};")
    return cursor.fetchall()

def empty_docs(workspace='./databases/', ext='.db'):
    files = find_docs(workspace, ext)

    if ext == '.db':
        for file in files:
            conn = connect(file)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            for table in tables:
                table_name = table[0]
                cursor.execute(f"DROP TABLE IF EXISTS `{table_name}`;")
                conn.commit()
            conn.close()
    elif ext == '.bin':
        for file in files:
            with open(file, "w") as f:
                f.write('')

def separate_docs(rows):
    embeddings, blockdata = [], []
    for row in rows:
        data = encoded_json(row[-1])
        embeddings.append(data["embedding"])
        blockdata.append(data)
    return embeddings, blockdata


def find_docs(path, ext='.db'):
    docs = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(ext):
                docs.append(os.path.join(root, file))
    return docs