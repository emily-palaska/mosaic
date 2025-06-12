import json

from llm_blockmerger.core import load_nb, load_py, encoded_json
from llm_blockmerger.load.code_loading import _preprocess_code_lines, _extract_cell_content

class CodeBlocksManager:
    def __init__(self, blocks=None, labels=None, source=None, variable_dictionaries=None):
        self.labels = labels if labels else []
        self.blocks = blocks if blocks else []
        self.variable_dictionaries = variable_dictionaries if variable_dictionaries is not None else []
        self.sources = source if source else []

    def __len__(self):
        return len(self.blocks)

    def __str__(self):
        # todo implement this
        return NotImplemented

    def preprocess_notebook(self, path, notebook):
        blocks, labels = _preprocess_code_lines(*_extract_cell_content(notebook))
        self.blocks, self.labels, self.sources = blocks, labels, path

    def preprocess_python_file(self, path, python_file):
        blocks, labels = _preprocess_code_lines([python_file], [''])
        self.blocks, self.labels, self.sources = blocks, labels, path

    def set(self, blocks=None, labels=None, source=None, variable_dictionaries=None):
        if blocks is not None: self.blocks = blocks
        if labels is not None: self.labels = labels
        if source is not None: self.sources = source
        if variable_dictionaries is not None: self.variable_dictionaries = variable_dictionaries

    def append_doc(self, doc):
        blockdata = encoded_json(doc.blockdata)
        self.blocks.append(blockdata["blocks"])
        self.labels.append(blockdata["label"])
        self.variable_dictionaries.append(blockdata["variable_dictionary"])
        if not isinstance(self.sources, list): self.sources = [self.sources]
        self.sources.append(blockdata["source"])

    def rearrange(self, order):
        assert len(order) == len(self.blocks), 'Inconsistent order'
        if self.blocks: self.blocks = [self.blocks[i] for i in order]
        if self.labels: self.labels = [self.labels[i] for i in order]
        self.variable_dictionaries = {k: v for d in self.variable_dictionaries for k, v in d.items()}
        if isinstance(self.sources, list) and self.sources: self.sources = [self.sources[i] for i in order]

    def unzip(self):
        return self.blocks, self.labels, self.variable_dictionaries, self.sources

def initialize_managers(paths):
    managers = []
    for path in paths:
        managers.append(CodeBlocksManager())
        if '.ipynb' in path: managers[-1].preprocess_notebook(*load_nb(path))
        elif '.py' in path: managers[-1].preprocess_python_file(*load_py(path))
        else: raise TypeError(f"Notebooks paths invalid datatype: {path}")
    return managers

def create_blockdata(block_managers, embeddings):
    total_blocks = sum(len(manager.blocks) for manager in block_managers)
    assert total_blocks == len(embeddings), f"{total_blocks} != {len(embeddings)}"
    embedding_iter = iter(embeddings.tolist())
    return [
        json.dumps({
            "label": manager.labels[i],
            "blocks": manager.blocks[i],
            "variable_dictionary": manager.variable_dictionaries[i],
            "source": manager.sources,
            "embedding": next(embedding_iter)
        })
        for manager in block_managers
        for i in range(len(manager))
    ]

def extract_labels(block_managers, blocks=False):
    if not blocks: return [label for block_manager in block_managers for label in block_manager.labels]

    symbols = ['!', '"', "'", ',', '.', ':', '-', '+', '=', '-', '>', '<', '(', ')', '[', ']', '{', '}']
    labels = [label for block_manager in block_managers for label in block_manager.labels]
    blocks = [block for block_manager in block_managers for block in block_manager.blocks]
    for block in blocks:
        for symbol in symbols:
            block.replace(symbol, ' ')
    return [label + '\nCODE:\n' + block for label, block in zip(labels, blocks)]

