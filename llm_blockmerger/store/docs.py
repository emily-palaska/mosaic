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

def empty_docs(workspace='./databases/'):
    files = find_docs(workspace)

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


def separate_docs(rows):
    embeddings, blockdata = [], []
    for row in rows:
        data = encoded_json(row[-1])
        embeddings.append(data["embedding"])
        blockdata.append(data)
    return embeddings, blockdata


def find_docs(path):
    files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.db'):
                files.append(os.path.join(root, file))
    return files

def print_docs(workspace="./databases/"):
    """ Function for debugging purposes only"""
    db_files = find_docs(workspace)
    print(f'Found files: {db_files}')
    for db_file in db_files:
        full_path = os.path.join(workspace, str(db_file))
        db_size = os.path.getsize(full_path)
        print(f"\nContents of database {db_file} with size {db_size/1024:.2f} KB")

        conn = sqlite3.connect(full_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table[0]

            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            row_count = cursor.fetchone()[0]

            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            column_count = len(columns)

            print(f"\nTable: {table_name}")
            print(f"Rows: {row_count}, Columns: {column_count}")

            column_names = [col[1] for col in columns]
            print("Columns:", ", ".join(column_names))

            cursor.execute(f"SELECT * FROM {table_name};")
            rows = cursor.fetchall()
            for row in rows:
                for element in row:
                    print(element)
                    print('-'*60)
                print()

        conn.close()