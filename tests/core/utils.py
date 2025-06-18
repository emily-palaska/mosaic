from llm_blockmerger.load import BlockManager
from textwrap import fill

def synthesis_dumb(synthesis: BlockManager, query:str, method: str, path: str):
    assert method in ['String', 'Embedding'], 'Invalid method'
    blocks, labels, var_dicts, sources = synthesis.unzip()

    content = f"# {method} Code Synthesis\nQuery: `{query}`\n"
    content += f"## Variables:\n"
    for key, value in var_dicts.items():
        content += f"- {key}:<br>\n>{value}\n"
    content += f"## Synthesis:\n"

    for block, label, source in zip(blocks, labels, sources):
        content += f'### {source}\n{fill(label, 150)}\n'
        content += f"```python\n{block}\n```\n\n"

    with open(path, 'w', encoding='utf-8') as file:
        file.write(content)