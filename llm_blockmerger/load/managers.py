from json import dumps
from textwrap import fill, indent

from llm_blockmerger.core import load_nb, load_py, encoded_json, remove_symbols
from llm_blockmerger.load.scripts import generate_blocks, cell_content

class BlockManager:
    def __init__(self, blocks=None, labels=None, source=None, var_dicts=None):
        self.labels = labels if labels else []
        self.blocks = blocks if blocks else []
        self.var_dicts = var_dicts if var_dicts is not None else []
        self.sources = source if source else []

    def __len__(self):
        return len(self.blocks)

    def __str__(self):
        divider = '=' * 60 + '\n'
        padding = '-' * 60 + '\n'
        string = "Source:" + str(self.sources) + '\n'

        for label, block, var_dict in zip(self.labels, self.blocks, self.var_dicts):
            string += padding
            string += fill(label, 80) + '\n'
            string += 'CODE:' + '\n'
            string += indent(block, '\t') + '\n'
            string += 'VARS:' + '\n'
            string += fill(dumps(var_dict, indent=2), 80) + '\n'
        return divider + string + divider

    def __getitem__(self, index):
        if isinstance(index, slice):
            return BlockManager(
                blocks=self.blocks[index],
                labels=self.labels[index],
                var_dicts=self.var_dicts[index],
                source=self.sources[index] if isinstance(self.sources, list) else self.sources
            )
        else:
            return BlockManager(
                blocks=[self.blocks[index]],
                labels=[self.labels[index]],
                var_dicts=[self.var_dicts[index]],
                source=[self.sources[index]] if isinstance(self.sources, list) else self.sources
            )

    def set(self, blocks=None, labels=None, source=None, var_dicts=None):
        if blocks is not None: self.blocks = blocks
        if labels is not None: self.labels = labels
        if source is not None: self.sources = source
        if var_dicts is not None: self.var_dicts = var_dicts

    def append_nb(self, path, nb):
        blocks, labels = generate_blocks(*cell_content(nb))
        self.blocks, self.labels, self.sources = blocks, labels, path
        self.var_dicts = [{} for _ in blocks]

    def append_py(self, path, file):
        blocks, labels = generate_blocks([file], [''])
        self.blocks, self.labels, self.sources = blocks, labels, path

    def append_doc(self, doc):
        blockdata = encoded_json(doc.blockdata)
        try: self.blocks.append(blockdata["block"])
        except KeyError: self.blocks.append(blockdata["blocks"])
        self.labels.append(blockdata["label"])
        self.var_dicts.append(blockdata["var_dict"])
        if not isinstance(self.sources, list): self.sources = [self.sources]
        self.sources.append(blockdata["source"])

    def rearrange(self, order):
        assert len(order) == len(self.blocks), f'Inconsistent order: {len(order)} != {len(self.blocks)}'
        if self.blocks: self.blocks = [self.blocks[i] for i in order]
        if self.labels: self.labels = [self.labels[i] for i in order]
        self.var_dicts = {k: v for d in self.var_dicts for k, v in d.items()}
        if isinstance(self.sources, list) and self.sources: self.sources = [self.sources[i] for i in order]

    def unzip(self):
        return self.blocks, self.labels, self.var_dicts, self.sources

def init_managers(paths):
    managers = []
    for path in paths:
        managers.append(BlockManager())
        if '.ipynb' in path: managers[-1].append_nb(path, load_nb(path))
        elif '.py' in path: managers[-1].append_py(path, load_py(path))
        else: raise TypeError(f"Notebooks paths invalid datatype: {path}")
    return managers

def create_blockdata(managers, embeddings):
    block_num = sum(len(manager.blocks) for manager in managers)
    assert block_num == len(embeddings), f"{block_num} != {len(embeddings)}"
    emb_iter = iter(embeddings.tolist())
    return [
        dumps({
            "label": manager.labels[i],
            "block": manager.blocks[i],
            "var_dict": manager.var_dicts[i],
            "source": manager.sources,
            "embedding": next(emb_iter)
        })
        for manager in managers
        for i in range(len(manager))
    ]

def flatten_labels(managers, code=False):
    if not code: return [label for block_manager in managers for label in block_manager.labels]

    labels = [label for block_manager in managers for label in block_manager.labels]
    blocks = [remove_symbols(block) for block_manager in managers for block in block_manager.blocks]

    return [label + '\nCODE:\n' + block for label, block in zip(labels, blocks)]

