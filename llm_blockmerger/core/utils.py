import ast, textwrap, os, json

def concatenate_block(block):
    return '\n'.join(block) + '\n'

def load_notebooks(nb_paths):
    if not isinstance(nb_paths, list): return os.path.basename(nb_paths), json.load(open(nb_paths, 'r'))
    return [(os.path.basename(path), json.load(open(path, 'r'))) for path in nb_paths]

def find_db_files(folder_path):
    db_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.db'):
                db_files.append(os.path.join(root, file))

    return db_files

def remove_common_words(original: str, to_remove: str, replacement='UNKNOWN') -> str:
    original = original.replace('\n', ' ')
    original_words = original.split()
    remove_words = set(word.lower() for word in to_remove.split())

    replaced_words = [
        replacement if word.lower() in remove_words else word
        for word in original_words
    ]
    return ' '.join(replaced_words)

def ast_extraction(script=''):
    tree = ast.parse(script)
    variables = set()

    def handle_function(curr_node):
        for arg in curr_node.args.args:
            variables.add(arg.arg)

    def handle_loop(curr_node):
        if isinstance(curr_node.target, ast.Name):
            variables.add(curr_node.target.id)
        elif isinstance(curr_node.target, ast.Tuple):
            handle_tuple(curr_node)

    def handle_tuple(curr_node):
        for target in curr_node.targets:
            if isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        variables.add(elt.id)

    def visit_node(curr_node):
        match curr_node:
            case ast.Name(ctx=ast.Store()):
                variables.add(curr_node.id)
            case ast.FunctionDef():
                handle_function(curr_node)
            case ast.For():
                handle_loop(curr_node)
            case ast.Assign():
                handle_tuple(curr_node)

    for node in ast.walk(tree):
        visit_node(node)

    return sorted(list(variables))

def print_merge_result(specification, block_manager):
    print("\n" + "=" * 60)
    print(' ' * 23 + "MERGE RESULT")
    print("=" * 60)

    print("\nSPECIFICATION:")
    print(textwrap.indent(specification, "    "))

    blocks, labels, variable_dictionaries, sources = block_manager.unzip()
    print("VARIABLES:")
    for block_dictionary in variable_dictionaries:
        for v, d in block_dictionary.items():
            print(f'\t{v}: {textwrap.fill(d,100)}')
    for i in range(len(block_manager)):
        print("-" * 60)
        print(f"SOURCE: {sources[i]}")
        print("LABEL:")
        print(textwrap.indent(textwrap.fill(labels[i],100), '\t'))
        print("CODE:")
        print(textwrap.indent(concatenate_block(blocks[i]), '\t'))

    print("\n" + "=" * 60)

def create_blockdata(labels, blocks, variable_dictionaries, sources):
    import json
    blockdata = [
        json.dumps(
            {
                'labels': labels[i],
                'blocks': blocks[i],
                'variable_dictionary': variable_dictionaries[i],
                'source': sources[i]
            }
        ) for i in range(len(blocks))
    ]
    return blockdata

def generate_triplets(n):
    from itertools import combinations
    return list(combinations(range(0, n), 3))

def print_db_contents(workspace='./databases/'):
    import sqlite3
    import os

    # Find all database files in the workspace
    db_files =  find_db_files(workspace)
    print(f'Found files: {db_files}')
    for db_file in db_files:
        full_path = os.path.join(workspace, db_file)
        db_size = os.path.getsize(full_path)
        print(f"\nContents of database {db_file} with size {db_size/1024:.2f} KB")


        conn = sqlite3.connect(full_path)
        cursor = conn.cursor()

        # Get all table names in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table[0]
            #cursor.execute(f"INSERT INTO {table_name} (label, data) VALUES ('John Doe', '555-1212');")

            # Get table size (approximate)
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            row_count = cursor.fetchone()[0]

            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            column_count = len(columns)

            print(f"\nTable: {table_name}")
            print(f"Rows: {row_count}, Columns: {column_count}")

            # Get and print column names
            column_names = [col[1] for col in columns]
            print("Columns:", ", ".join(column_names))

            # Print first few rows as sample
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
            rows = cursor.fetchall()
            print("\nSample rows:")
            for row in rows:
                for element in row:
                    print(element)
                    print('-'*60)
                print()

        conn.close()