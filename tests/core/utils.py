import numpy as np
from llm_blockmerger.load import BlockManager
from textwrap import fill
from re import sub, escape

def md_dumb_synthesis(synthesis: BlockManager, query:str, method: str, path: str):
    assert method in ['String', 'Embedding', 'Exhaustive', 'Reverse Embedding', 'Random'], f'Invalid method {method}'
    blocks, labels, var_dicts, sources = synthesis.unzip()

    content = f"# {method} Code Synthesis\nQuery `{query}`\n"
    content += f"## Script Variables\n"
    for key, value in var_dicts.items():
        content += f"- {key}:<br>\n>{value}\n"
    content += f"## Synthesis Blocks\n"

    for block, label, source in zip(blocks, labels, sources):
        content += f'### {source}\n{fill(label, 150)}\n'
        content += f"```python\n{block}\n```\n\n"

    content += "## Code Concatenation\n```python\n"
    for block in blocks: content += f'{block}\n'
    content += "```\n"

    with open(path, 'w', encoding='utf-8') as file:
        file.write(content)


def slice_2d(lists: list, limit: int):
    cut, count = [], 0
    for l in lists:
        take = min(len(l), max(0, limit - count))
        if take == 0: break
        cut.append(l[:take])
        count += take
    return cut


def linear_regression(x, y):
    if not isinstance(x, np.ndarray): x = np.array(x)
    if not isinstance(y, np.ndarray): y = np.array(y)

    x_mean, y_mean = np.mean(x), np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    m = numerator / denominator
    b = y_mean - m * x_mean

    y_pred = m * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return y_pred, m, b, r_squared


def separate_methods(synthesis_list):
    info = dict()
    for synthesis_dict in synthesis_list:
        for m, s in synthesis_dict.items():
            if m not in info: info[m] = {"codes":[], "blocks":[], "lines":[]}
            info[m]["codes"].append('\n'.join(s.labels) + '\n' + '\n'.join(s.blocks))
            #info[m]["codes"].append('\n'.join(s.blocks))
            info[m]["blocks"].append(len(s))
            info[m]["lines"].append(sum(len(block.splitlines()) for block in s.blocks))
    return info


def remove_words(queries: list, stopwords: list):
    pattern = r'\b(?:' + '|'.join(map(escape, stopwords)) + r')\b'
    return [sub(pattern, '', q) for q in queries]