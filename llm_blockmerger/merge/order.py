from llm_blockmerger.load import BlockManager
from llm_blockmerger.core import ast_io_split

def io_order(io_splits: list):
    outputs = set()
    outputs.add(out_var for io_split in io_splits for out_var in io_split['output'])

    order, def_outputs = [], set()
    remaining = set(range(len(io_splits)))
    while remaining:
        min_violations, min_index = float('inf'), 0
        for i in list(remaining):
            violations = len(io_splits[i]['input'] - def_outputs)
            if violations < min_violations:
                min_violations = violations
                min_index = i

        order.append(min_index)
        def_outputs |= io_splits[min_index]['output']
        remaining.remove(min_index)
    return order


def import_order(synthesis: BlockManager):
    return [i for i, block in enumerate(synthesis.blocks) if 'import' in block]


def synthesis_order(synthesis: BlockManager):
    io = io_order(ast_io_split(synthesis))
    imp = import_order(synthesis)
    for i in imp: io.remove(i)
    return imp + io