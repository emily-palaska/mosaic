from llm_blockmerger.core import ast_io_split
from llm_blockmerger.load import CodeBlocksManager

def cumulative_io_split(manager:CodeBlocksManager):
    return [
        ast_io_split(script=block, variables=var_dict)
            for block, var_dict in zip(manager.blocks, manager.variable_dictionaries)
    ]


def find_block_order(io_splits: list):
    all_outputs = set()
    all_outputs.add(output_var for io_split in io_splits for output_var in io_split['output'])

    order, defined_outputs = [], set()
    remaining = set(range(len(io_splits)))
    while remaining:
        min_violations, min_index = float('inf'), 0
        for i in list(remaining):
            violations = len(io_splits[i]['input'] - defined_outputs)
            if violations < min_violations:
                min_violations = violations
                min_index = i

        order.append(min_index)
        defined_outputs |= io_splits[min_index]['output']
        remaining.remove(min_index)
    return order