from llm_blockmerger.core import ast_io_split
from llm_blockmerger.load import BlockManager

def var_split(manager:BlockManager):
    return [
        ast_io_split(script=block, variables=var_dict)
            for block, var_dict in zip(manager.blocks, manager.var_dicts)
    ]


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