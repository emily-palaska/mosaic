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
        md = accumulated_markdown[i].replace('#', '')
        current_block, current_label = [], ''

        for line in code:
            if '#' in line:  # Comment detected
                if not line.lstrip().startswith('#'):  # Side-comment
                    before_hash, after_hash = line.split('#', 1)
                    blocks.append([before_hash])
                    labels.append(f"MARKDOWN: {md}\nCOMMENT: {after_hash.strip()}".replace('```', ''))
                    current_block.append(before_hash)
                else:  # Full-line
                    if current_block:
                        blocks.append(current_block if isinstance(current_block, list) else [current_block])
                        labels.append(f"MARKDOWN: {md}\nCOMMENT: {current_label}".replace('```', ''))
                    _, current_label = line.split('#', 1)
                    current_block = []
            else:  # Code without comment
                current_block.append(line)

        if current_block:  # Handle remaining lines in the current block
            blocks.append(current_block)
            labels.append(f"MARKDOWN: {md}\nCOMMENT: {current_label.strip()}".replace('```', ''))
    return blocks, labels