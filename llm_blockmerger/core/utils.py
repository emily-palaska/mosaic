import textwrap, os, json, re

def concatenate_block(block):
    return '\n'.join(block) + '\n'

def load_notebooks(ipynb_paths):
    if isinstance(ipynb_paths, str): return os.path.basename(ipynb_paths), json.load(open(ipynb_paths, 'r'))
    return [(os.path.basename(path), json.load(open(path, 'r'))) for path in ipynb_paths]

def load_python_files(py_paths):
    if isinstance(py_paths, str): return os.path.basename(py_paths), [line for line in open(py_paths, 'r', encoding='utf-8')]
    return [(os.path.basename(path), [line for line in open(path, 'r', encoding='utf-8')]) for path in py_paths]

def remove_common_words(original: str, to_remove: str, replacement='UNKNOWN') -> str:
    original = original.replace('\n', ' ')
    original_words = original.split()
    remove_words = set(word.lower() for word in to_remove.split())

    replaced_words = [
        replacement if word.lower() in remove_words else word
        for word in original_words
    ]
    return ' '.join(replaced_words)


def remove_common_indentation(blocks):
    unintended_blocks = []
    for block in blocks:
        lines = block.splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        indentations = [
            len(re.match(r'^\s*', line).group())
                for line in non_empty_lines
        ]

        if not indentations:
            unintended_blocks.append(block)
            continue
        common_indent = min(indentations)

        unindented_lines = []
        for line in lines:
            unindented_lines.append(line[common_indent:] if line.strip() else '')
        unintended_blocks.append('\n'.join(unindented_lines))
    return unintended_blocks

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

def print_managers(block_managers):
    print('=' * 55)
    for manager in block_managers:
        for label, block in zip(manager.labels, manager.blocks):
            print(textwrap.fill(label, 80) + '\n')
            print(textwrap.indent(block, '\t'))
            print('-' * 60)
    print('=' * 55)

