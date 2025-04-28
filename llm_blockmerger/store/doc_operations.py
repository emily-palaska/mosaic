import sqlite3, os, json
from docarray import BaseDoc
from docarray.typing import TorchTensor

def find_db_files(folder_path):
    db_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.db'):
                db_files.append(os.path.join(root, file))
    return db_files

def make_doc(feature_size=384):
    class BlockMergerDoc(BaseDoc):
        id: str = ''
        embedding: TorchTensor[feature_size]
        blockdata: str = ''

    return BlockMergerDoc

def generate_triplets(n):
    from itertools import combinations
    return list(combinations(range(0, n), 3))

def get_db_rows(db_file_path:str, table_name='docs'):
    import sqlite3
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name};")
    return cursor.fetchall()

def extract_rows_content(rows):
    embeddings, blockdata = [], []
    for row in rows:
        print(row[-1])
        data = json.loads(row[-1])
        print(type(data))
        embeddings.append(data['embedding'])
        blockdata.append(data)
    return embeddings, blockdata

def empty_docs(workspace='./databases/'):
    db_files = find_db_files(workspace)

    for db_file in db_files:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table[0]
            cursor.execute(f"DROP TABLE IF EXISTS `{table_name}`;")
            conn.commit()
        conn.close()

def print_db_contents(workspace="./databases/"):
    """ Function for debugging purposes only"""
    db_files =  find_db_files(workspace)
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