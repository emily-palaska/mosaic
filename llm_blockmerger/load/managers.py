from llm_blockmerger.core.utils import load_notebooks
from llm_blockmerger.load.code_loading import _preprocess_code_lines, _extract_cell_content

class CodeBlocksManager:
    def __init__(self, blocks=None, labels=None, source=None, variable_dictionaries=None):
        self.labels = labels if labels else []
        self.blocks = blocks if blocks else []
        self.variable_dictionaries = variable_dictionaries if variable_dictionaries else []
        self.sources = source if source else []

    def __len__(self):
        return len(self.blocks)

    def preprocess_notebook(self, path, notebook):
        blocks, labels = _preprocess_code_lines(*_extract_cell_content(notebook))
        self.blocks = blocks
        self.labels = labels
        self.sources = path

    def set(self, blocks=None, labels=None, source=None, variable_dictionaries=None):
        if blocks is not None: self.blocks = blocks
        if labels is not None: self.labels = labels
        if source is not None: self.sources = source
        if variable_dictionaries is not None: self.variable_dictionaries = variable_dictionaries

    def append_doc(self, doc):
        self.blocks.append(doc.block)
        self.labels.append(doc.label)
        self.variable_dictionaries.append(doc.variable_dictionary)
        if not isinstance(self.sources, list): self.sources = [self.sources]
        self.sources.append(doc.source)

    def unzip(self):
        return self.blocks, self.labels, self.variable_dictionaries, self.sources

def initialize_managers(notebook_paths):
    managers = []
    for notebook_path in notebook_paths:
        managers.append(CodeBlocksManager())
        managers[-1].preprocess_notebook(*load_notebooks(notebook_path))
    return managers

def concatenate_managers(block_managers):
    labels, blocks, sources, variable_dictionaries = [], [], [], []
    for block_manager in block_managers:
        labels.extend(block_manager.labels)
        blocks.extend(block_manager.blocks)
        variable_dictionaries.extend(block_manager.variable_dictionaries)
        sources.extend(block_manager.sources for _ in range(len(block_manager)))
    return blocks, labels, variable_dictionaries, sources
