import ast, re
from llm_blockmerger.core import dedent_blocks

class VariableAnalyzer(ast.NodeVisitor):
    def __init__(self, io_split=False):
        self.io_split = io_split
        self.variables = set()
        self.read = set()
        self.written = set()

    def visit_Name(self, node):
        if self.io_split:
            if isinstance(node.ctx, ast.Load):
                self.read.add(node.id)
            elif isinstance(node.ctx, ast.Store):
                self.written.add(node.id)
        self.variables.add(node.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        for arg in node.args.args:
            self.variables.add(arg.arg)
        self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            self._recursive_tuple(target)
        self.generic_visit(node)

    def visit_For(self, node):
        self._recursive_tuple(node.target)
        self.generic_visit(node)

    def _recursive_tuple(self, target):
        if isinstance(target, ast.Name):
            self.variables.add(target.id)
        elif isinstance(target, ast.Tuple):
            for elt in target.elts:
                self._recursive_tuple(elt)  # Recursively handle nested tuples

def parse_script(script:str):
    while script != '':
        try:
            tree = ast.parse(script)
            return tree
        except (IndentationError, SyntaxError):
            script = dedent_blocks(script.split('\n', 1)[1])[0]
    return None

def ast_extraction(script: str):
    analyzer, tree = VariableAnalyzer(), parse_script(script)
    if tree is None: return []
    analyzer.visit(tree)
    return sorted(list(analyzer.variables))

def ast_io_split(variables: dict, script: str):
    analyzer, tree = VariableAnalyzer(io_split=True), parse_script(script)
    if tree is None: return {'input': set(), 'output': set()}
    analyzer.visit(tree)

    input_vars, output_vars = set(), set()
    for var in variables:
        is_read = var in analyzer.read
        is_written = var in analyzer.written

        if is_read and not is_written: input_vars.add(var)
        elif is_written: output_vars.add(var)

    return {'input': input_vars, 'output': output_vars}

def var_separation(blocks, nb_vars):
    var_dicts = []
    for block in blocks:
        block_dict = {}
        for var, desc in nb_vars:
            if re.search(rf'\b{var}\b', block):
                block_dict[var] = desc
        var_dicts.append(block_dict)
    return var_dicts

def parse_vars(output):
    if "Output:" in output:
        var_str = output.split("Output:")[1].strip()
        var_str = var_str.split("\n")[0].strip()
        var_set = {
            var.strip().replace("[", "").replace("]", "")
            for var in var_str.split(",")
        }
        var_set.discard("")
    else:
        var_set = set()
    return sorted(var_set)

def parse_desc(output):
    if "Output:" in output:
        description_str = output.split("Output:")[1].strip()
        description_str = description_str.split(":")[1].strip()
        description_str = description_str.split("\n")[0].strip()
    else:
        description_str = "LLM failed to generate description that fits the predefined structure."
    return description_str