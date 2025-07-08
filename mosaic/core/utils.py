from json import load, loads
from re import match, sub, escape
from textwrap import indent, fill

from torch import Tensor, tensor
from itertools import combinations


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
    if not isinstance(blocks, list): blocks = [blocks]
    dedented_blocks = []
    for block in blocks:
        lines = block if isinstance(block, list) else block.splitlines()
        script_lines = [line for line in lines if line.strip()]
        ind = [len(match(r'^\s*', line).group())for line in script_lines]

        if not ind:
            dedented_blocks.append(lines)
            continue
        common_ind = min(ind)

        dedented_lines = [line[common_ind:] if line.strip() else '' for line in lines]
        dedented_blocks.append('\n'.join(dedented_lines))
    if dedented_blocks and len(dedented_blocks) > 1: return dedented_blocks
    elif dedented_blocks and len(dedented_blocks) == 1: return dedented_blocks[0]
    else: return ''


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


def best_combination(n: int, r: int, target: float, components: Tensor, batch_size: int = 1.0e8):
    best_sum, best_comb = None, None
    device, batch = components.device, []

    for comb in combinations(range(n), r):
        batch.append(comb)
        if len(batch) == batch_size:
            best_sum, best_comb = _process_batch(batch, components, target, best_sum, best_comb)
            batch = []

    if batch: best_sum, best_comb = _process_batch(batch, components, target, best_sum, best_comb)
    return best_sum, best_comb


def _process_batch(batch: list, components: Tensor, target: float, best_sum: float, best_comb: Tensor):
    comb_tensor = tensor(batch, device=components.device)
    selected = components[comb_tensor]
    sums = selected.sum(dim=1)
    valid = (sums <= target)
    if valid.any():
        sums = sums[valid]
        max_sum, max_idx = sums.max(0)
        if best_sum is None or max_sum > best_sum: return max_sum.item(), comb_tensor[valid][max_idx]
    return best_sum, best_comb


def separate_lines(source):
    separated = []
    for string in source:
        if '\n' in string: separated.extend([line for line in string.split('\n') if line != ''])
        else: separated.append(string)
    return separated


def regular_replace(string, old, new):
    return sub(r'\b' + escape(old) + r'\b', new, string)
