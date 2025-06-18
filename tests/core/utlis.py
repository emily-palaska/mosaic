from llm_blockmerger.load import BlockManager

def synthesis_dumb(synthesis: BlockManager, query:str, method: str, path: str):
    assert method in ['String', 'Embedding'], 'Invalid method'
    content = f"# {method} Code Synthesis\nQuery: `{query}`"

    for block, label, var_dict, source in synthesis.unzip():
        content += f'## {source}\n{label}\n'
        content += f"```python\n{var_dict}\n```\n"
        content += f"```python{block}\n```\n\n"

    with open(path, 'w', encoding='utf-8') as file:
        file.write(content)