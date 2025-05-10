import ast, re

def ast_extraction(script=''):
    tree = ast.parse(script)
    variables = set()

    def handle_function(curr_node):
        for arg in curr_node.args.args:
            variables.add(arg.arg)

    def handle_loop(curr_node):
        if isinstance(curr_node.target, ast.Name):
            variables.add(curr_node.target.id)
        elif isinstance(curr_node.target, ast.Tuple):
            handle_tuple(curr_node)

    def handle_tuple(curr_node):
        for target in curr_node.targets:
            if isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        variables.add(elt.id)

    def visit_node(curr_node):
        match curr_node:
            case ast.Name(ctx=ast.Store()):
                variables.add(curr_node.id)
            case ast.FunctionDef():
                handle_function(curr_node)
            case ast.For():
                handle_loop(curr_node)
            case ast.Assign():
                handle_tuple(curr_node)

    for node in ast.walk(tree):
        visit_node(node)

    return sorted(list(variables))

# todo this should be integrated to ast extraction
class VariableAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.read = set()
        self.written = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.read.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.written.add(node.id)
        self.generic_visit(node)

def split_io_variables(variables: dict, script: str):
    analyzer = VariableAnalyzer()
    analyzer.visit(ast.parse(script))
    input_vars, output_vars = set(), set()

    for var in variables:
        is_read = var in analyzer.read
        is_written = var in analyzer.written

        if is_read and not is_written: input_vars.add(var)
        elif is_written and not is_read: output_vars.add(var)

    return {'input': input_vars, 'output': output_vars}

def find_block_order(blocks:list, var_dicts:list):
    var_splits, all_outputs = [], set()
    for i in range(len(blocks)):
        var_split = split_io_variables(var_dicts[i], blocks[i])
        var_splits.append(var_split)
        all_outputs |= var_split['output']

    order, used_outputs = [], set()
    remaining = set(range(len(blocks)))
    while remaining:
        scheduled = False
        for i in list(remaining):
            if var_splits[i]['input'].issubset(used_outputs):
                order.append(i)
                used_outputs |= var_splits[i]['output']
                remaining.remove(i)
                scheduled = True
                break
        if not scheduled: # Fallback: pick one block to reduce deadlock
            i = remaining.pop()
            order.append(i)
            used_outputs |= var_splits[i]['output']

    return order

def separate_variables_per_block(blocks, notebook_variables):
    variable_dictionaries = []
    for block in blocks:
        block_dictionary = {}
        for variable, description in notebook_variables:
            if re.search(rf'\b{variable}\b', block):
                block_dictionary[variable] = description
        variable_dictionaries.append(block_dictionary)
    return variable_dictionaries

def separate_variable_string(output_text):
    if "Output:" in output_text:
        variables_str = output_text.split("Output:")[1].strip()
        variables_str = variables_str.split("\n")[0].strip()
        variables = {
            var.strip().replace("[", "").replace("]", "")
            for var in variables_str.split(",")
        }
        variables.discard("")
    else:
        variables = set()
    return sorted(variables)

def separate_description_string(output_text):
    if "Output:" in output_text:
        description_str = output_text.split("Output:")[1].strip()
        description_str = description_str.split(":")[1].strip()
        description_str = description_str.split("\n")[0].strip()
    else:
        description_str = "LLM failed to generate description that fits the predefined structure."
    return description_str