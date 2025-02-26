import json
from loading.variable_extracting import extract_variables

class CodeBlocksManager:
    def __init__(self, blocks=None, labels=None, variables=None, var_descriptions=None):
        if blocks is None:
            blocks = []
        self.labels = labels if labels else []
        self.blocks = blocks if blocks else []
        self.variables = variables if variables else set()
        self.var_descriptions = var_descriptions if var_descriptions else []

def load_notebooks(nb_paths):
    return [json.load(open(path, 'r')) for path in nb_paths]

def _extract_cell_content(notebook):
    code_lines, accumulated_markdown = [], []
    cell_markdown = ''

    for cell in notebook.get('cells', []):
        if cell['cell_type'] == 'markdown':
            cell_markdown += ''.join(cell['source'])
        elif cell['cell_type'] == 'code':
            code_lines.append(cell['source'])
            accumulated_markdown.append(cell_markdown  if cell_markdown else accumulated_markdown[-1])
            cell_markdown = ''
    return code_lines, accumulated_markdown

def _preprocess_code_lines(code_lines, accumulated_markdown):
    blocks, labels = [], []

    for i, code in enumerate(code_lines):
        md = accumulated_markdown[i]
        current_block, current_label = [], ''

        for line in code:
            if '#' in line:  # Comment detected
                if not line.startswith('#'):  # Side-comment
                    before_hash, after_hash = line.split('#', 1)
                    blocks.append([before_hash])
                    labels.append(f"MARKDOWN: {md}\nCOMMENT: {after_hash.strip()}")
                    current_block.append(before_hash)
                else:  # Full-line
                    if current_block:
                        blocks.append(current_block if isinstance(current_block, list) else [current_block])
                        labels.append(f"MARKDOWN: {md}\nCOMMENT: {current_label}")
                    _, current_label = line.split('#', 1)
                    current_block = []
            else:  # Code without comment
                current_block.append(line)

        if current_block:  # Handle remaining lines in the current block
            blocks.append(current_block)
            labels.append(f"MARKDOWN: {md}\nCOMMENT: {current_label.strip()}")
    return blocks, labels

def preprocess_blocks(notebook_data):
    block_managers = []
    for notebook in notebook_data:
        code_lines, accumulated_markdown = _extract_cell_content(notebook)
        block_managers.append(CodeBlocksManager(*_preprocess_code_lines(code_lines, accumulated_markdown)))
        for block in block_managers[-1].blocks:
            try: block_managers[-1].variables.update(extract_variables(_concatenate_block(block)))
            except IndentationError: continue
            except Exception: raise
    return block_managers

def _concatenate_block(block):
    return '\n'.join(block) + '\n'

def concatenate_managers(block_managers):
    labels, blocks, variables = [], [], []
    for block_manager in block_managers:
        labels.extend(block_manager.labels)
        blocks.extend(block_manager.blocks)
        variables.append(block_manager.variables)
    return blocks, labels, variables