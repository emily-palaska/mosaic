from json import load, loads
from re import match
from textwrap import indent, fill

from torch import Tensor, combinations, arange


def concat_block(block):
    return '\n'.join(block) + '\n'

def load_nb(paths):
    if isinstance(paths, str): return load(open(paths, 'r'))
    return [load(open(path, 'r')) for path in paths]

def load_py(paths):
    if isinstance(paths, str): return open(paths, 'r', encoding='utf-8')
    return [[line for line in open(path, 'r', encoding='utf-8')] for path in paths]

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

def dedent_blocks(blocks):
    dedented_blocks = []
    for block in blocks:
        lines = block.splitlines()
        script_lines = [line for line in lines if line.strip()]
        ind = [
            len(match(r'^\s*', line).group())
                for line in script_lines
        ]

        if not ind:
            dedented_blocks.append(block)
            continue
        common_ind = min(ind)

        dedented_lines = [line[common_ind:] if line.strip() else '' for line in lines]
        dedented_blocks.append('\n'.join(dedented_lines))
    return dedented_blocks

def encoded_json(field: str):
    loaded_once = loads(field)
    if isinstance(loaded_once, str):
        return loads(loaded_once)
    return loaded_once

def print_synthesis(manager, specification: str, title='STRING'):
    print("\n" + "=" * 60)
    print(' ' * 15 + f"MERGE RESULT ({title})")
    print("=" * 60)

    print("\nSPECIFICATION:")
    print(indent(specification, "    "))

    blocks, labels, variable_dictionaries, sources = manager.unzip()
    print("VARIABLES:")
    for v, d in manager.var_dicts.items():
        print(f'\t{v}: {fill(d,80)}')
    for i in range(len(manager)):
        print("-" * 60)
        print(f"SOURCE: {sources[i]}")
        print("LABEL:")
        print(indent(fill(labels[i],80), '\t'))
        print("CODE:")
        print(indent(blocks[i], '\t'))

    print("\n" + "=" * 60)


def triplets(n):
    from itertools import combinations
    return list(combinations(range(0, n), 3))

def remove_symbols(script):
    symbols = ['!', '"', "'", ',', '.', ':', '-', '+', '=', '-', '>', '<', '(', ')', '[', ']', '{', '}']
    for symbol in symbols:
        script = script.replace(symbol, ' ')
    return script


def best_combination(n: int, r: int, target: float, components: Tensor):
    device = components.device
    comb = combinations(arange(n, device=device), r=r)
    selected = components[comb]
    sums = selected.sum(dim=1)

    valid = (sums <= target)
    if valid.any():
        sums = sums[valid]
        max_sum, max_idx = sums.max(0)
        return max_sum, comb[valid][max_idx]
    return None, None
