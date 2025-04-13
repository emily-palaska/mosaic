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

    blocks, labels, variables, descriptions, sources = block_manager.unzip()
    print("VARIABLES:")
    for i in range(len(block_manager)):
        for v, d in zip(variables[i], descriptions[i]):
            print(f'\t{v}: {textwrap.fill(d,100)}')
    for i in range(len(block_manager)):
        print("-" * 60)
        print(f"SOURCE: {sources[i]}")
        print("LABEL:")
        print(textwrap.indent(textwrap.fill(labels[i],100), '\t'))
        print("CODE:")
        print(textwrap.indent(concatenate_block(blocks[i]), '\t'))

    print("\n" + "=" * 60)