from os.path import basename
from json import load, loads
from re import match
from textwrap import indent, fill

def concat_block(block):
    return '\n'.join(block) + '\n'

def load_nb(paths):
    if isinstance(paths, str): return basename(paths), load(open(paths, 'r'))
    return [(basename(path), load(open(path, 'r'))) for path in paths]

def load_py(paths):
    if isinstance(paths, str): return basename(paths), [line for line in open(paths, 'r', encoding='utf-8')]
    return [(basename(path), [line for line in open(path, 'r', encoding='utf-8')]) for path in paths]

def remove_common_words(og: str, rem: str, repl='[UNK]'):
    og_words = og.replace('\n', ' ').split()
    rem_words = set(word.lower() for word in rem.split())

    repl_words = []
    for word in og_words:
        word = word.lower()
        if word in rem_words:
            repl_words.append(repl.lower())
            rem_words.remove(word)
        else: repl_words.append(word)

    return ' '.join(repl_words)

def remove_common_indentation(blocks):
    unintended_blocks = []
    for block in blocks:
        lines = block.splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        indentations = [
            len(match(r'^\s*', line).group())
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

def encoded_json(field: str):
    loaded_once = loads(field)
    if isinstance(loaded_once, str):
        return loads(loaded_once)
    return loaded_once

def print_merge_result(specification, manager, title='STRING'):
    print("\n" + "=" * 60)
    print(' ' * 15 + f"MERGE RESULT ({title})")
    print("=" * 60)

    print("\nSPECIFICATION:")
    print(indent(specification, "    "))

    blocks, labels, variable_dictionaries, sources = manager.unzip()
    print("VARIABLES:")
    for v, d in manager.variable_dictionaries.items():
        print(f'\t{v}: {fill(d,80)}')
    for i in range(len(manager)):
        print("-" * 60)
        print(f"SOURCE: {sources[i]}")
        print("LABEL:")
        print(indent(fill(labels[i],80), '\t'))
        print("CODE:")
        print(indent(blocks[i], '\t'))

    print("\n" + "=" * 60)

def print_managers(block_managers):
    print('=' * 55)
    for manager in block_managers:
        for label, block in zip(manager.labels, manager.blocks):
            print(fill(label, 80) + '\n')
            print(indent(block, '\t'))
            print('-' * 60)
    print('=' * 55)

def triplets(n):
    from itertools import combinations
    return list(combinations(range(0, n), 3))
