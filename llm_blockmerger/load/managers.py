from llm_blockmerger.core.utils import load_notebooks
from llm_blockmerger.load.code_loading import _preprocess_code_lines, _extract_cell_content

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

def initialize_managers(notebook_paths):
    managers = []
    for notebook_path in notebook_paths:
        managers.append(CodeBlocksManager())
        managers[-1].preprocess_notebook(*load_notebooks(notebook_path))
    return managers

def concatenate_managers(block_managers):
    labels, blocks, sources, variables, var_descriptions = [], [], [], [], []
    for block_manager in block_managers:
        labels.extend(block_manager.labels)
        blocks.extend(block_manager.blocks)
        variables.extend(block_manager.variables)
        sources.extend(block_manager.sources for _ in range(len(block_manager)))
        var_descriptions.extend(block_manager.var_descriptions)
    return blocks, labels, variables, var_descriptions, sources
