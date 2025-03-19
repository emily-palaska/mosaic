class CodeBlocksManager:
    def __init__(self, blocks=None, labels=None, source=None, variables=None, var_descriptions=None):
        self.labels = labels if labels else []
        self.blocks = blocks if blocks else []
        self.variables = variables if variables else []
        self.var_descriptions = var_descriptions if var_descriptions else []
        self.sources = source if source else []

    def __len__(self):
        return len(self.blocks)

    def preprocess_notebook(self, path, notebook):
        blocks, labels = _preprocess_code_lines(*_extract_cell_content(notebook))
        self.blocks = blocks
        self.labels = labels
        self.sources = path

    def set(self, blocks=None, labels=None, source=None, variables=None, var_descriptions=None):
        if blocks is not None: self.blocks = blocks
        if labels is not None: self.labels = labels
        if source is not None: self.sources = source
        if variables is not None: self.variables = variables
        if var_descriptions is not None: self.var_descriptions = var_descriptions

    def append_doc(self, doc):
        self.blocks.append(doc.block)
        self.labels.append(doc.label)
        self.variables.append(doc.variables)
        self.var_descriptions.append(doc.var_descriptions)
        if not isinstance(self.sources, list): self.sources = [self.sources]
        self.sources.append(doc.source)

    def unzip(self):
        return self.blocks, self.labels, self.variables, self.var_descriptions, self.sources

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
"""
# alternative function to CodeBlocksManager.preprocess_notebook
def preprocess_blocks(notebook_data):
    block_managers = []
    for path, notebook in notebook_data:
        code_lines, accumulated_markdown = _extract_cell_content(notebook)
        block_managers.append(CodeBlocksManager(*_preprocess_code_lines(code_lines, accumulated_markdown), path))
    return block_managers

"""