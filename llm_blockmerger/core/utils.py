import ast, textwrap, os, json

def concatenate_block(block):
    return '\n'.join(block) + '\n'

def load_notebooks(ipynb_paths):
    if isinstance(ipynb_paths, str): return os.path.basename(ipynb_paths), json.load(open(ipynb_paths, 'r'))
    return [(os.path.basename(path), json.load(open(path, 'r'))) for path in ipynb_paths]

def load_python_files(py_paths):
    if isinstance(py_paths, str): return [line for line in open(py_paths, 'r', encoding='utf-8')]
    return [[line for line in open(path, 'r', encoding='utf-8')] for path in py_paths]

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

def load_double_encoded_json(field: str):
    loaded_once = json.loads(field)
    if isinstance(loaded_once, str):
        return json.loads(loaded_once)
    return loaded_once

def print_merge_result(specification, block_manager, merge_type='STRING'):
    print("\n" + "=" * 60)
    print(' ' * 15 + f"MERGE RESULT ({merge_type})")
    print("=" * 60)

    print("\nSPECIFICATION:")
    print(textwrap.indent(specification, "    "))

    blocks, labels, variable_dictionaries, sources = block_manager.unzip()
    print("VARIABLES:")
    for v, d in block_manager.variable_dictionaries.items():
        print(f'\t{v}: {textwrap.fill(d,80)}')
    for i in range(len(block_manager)):
        print("-" * 60)
        print(f"SOURCE: {sources[i]}")
        print("LABEL:")
        print(textwrap.indent(textwrap.fill(labels[i],80), '\t'))
        print("CODE:")
        print(textwrap.indent(blocks[i], '\t'))

    print("\n" + "=" * 60)

