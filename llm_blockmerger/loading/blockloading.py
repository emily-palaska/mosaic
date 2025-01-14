import json

def load_notebooks(nb_paths):
    return [json.load(open(path, 'r')) for path in nb_paths]

def extract_cell_content(notebook):
    code_lines, accumulated_markdown = [], []
    cell_markdown = ''

    for cell in notebook.get('cells', []):
        if cell['cell_type'] == 'markdown':
            cell_markdown += ''.join(cell['source'])
        elif cell['cell_type'] == 'code':
            code_lines.append(cell['source'])
            accumulated_markdown.append(cell_markdown or accumulated_markdown[-1] if accumulated_markdown else '')
            cell_markdown = ''
    return code_lines, accumulated_markdown


def separate_blocks(notebook_data):
    all_code_lines, all_accumulated_markdown = [], []
    for notebook in notebook_data:
        code_lines, accumulated_markdown = extract_cell_content(notebook)
        all_code_lines.extend(code_lines)
        all_accumulated_markdown.extend(accumulated_markdown)
    return all_code_lines, all_accumulated_markdown


def parse_code_blocks(code_lines, accumulated_markdown):
    # TODO deal with 3 "s type of comments
    blocks, labels = [], []

    for i, code in enumerate(code_lines):
        md = accumulated_markdown[i]
        current_block, current_label = [], ''

        for line in code:
            if '#' in line:  # Comment detected
                if not line.startswith('#'): # Side-comment
                    before_hash, after_hash = line.split('#', 1)
                    blocks.append([before_hash.strip()])
                    labels.append(f"MARKDOWN: {md}\nCOMMENT: {after_hash.strip()}")
                else: # Full-line
                    if current_block:
                        blocks.append(current_block if isinstance(current_block, list) else [current_block])
                        labels.append(f"MARKDOWN: {md}\nCOMMENT: {current_label}")
                    _, current_label = line.split('#', 1)
                    current_block = []
            else:  # Code without comment
                current_block.append(line.strip())

        if current_block:  # Handle remaining lines in the current block
            blocks.append(current_block)
            labels.append(f"MARKDOWN: {md}\nCOMMENT: {current_label.strip()}")

    return blocks, labels

def preprocess_blocks(code_lines, accumulated_markdown):
    return parse_code_blocks(code_lines, accumulated_markdown)
