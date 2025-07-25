import ast
from mosaic.core import dedent_blocks

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
            new_script = script.split('\n', 1)
            script = dedent_blocks(new_script[1]) if len(new_script) > 1 else ''
        except TypeError:
            break
    return None

def ast_extraction(script: str):
    analyzer, tree = VariableAnalyzer(), parse_script(script)
    if tree is None: return []
    analyzer.visit(tree)
    return sorted(list(analyzer.variables))

def ast_io_split(manager):
    io_splits = []
    for script, variables in zip(manager.blocks, manager.var_dicts):
        analyzer, tree = VariableAnalyzer(io_split=True), parse_script(script)
        if tree is None:
            io_splits.append({'input': set(), 'output': set()})
            continue
        analyzer.visit(tree)

        input_vars, output_vars = set(), set()
        for var in variables:
            is_read = var in analyzer.read
            is_written = var in analyzer.written

            if is_read and not is_written: input_vars.add(var)
            elif is_written: output_vars.add(var)
        io_splits.append({'input': input_vars, 'output': output_vars})
    return io_splits