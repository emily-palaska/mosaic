import ast

def ast_extract_variables(script):
    tree = ast.parse(script)
    variables = set()

    def visit_node(curr_node):
        if isinstance(curr_node, ast.Name) and isinstance(curr_node.ctx, ast.Store):
            variables.add(curr_node.id)
        elif isinstance(curr_node, ast.FunctionDef):
            # Capture function arguments
            for arg in curr_node.args.args:
                variables.add(arg.arg)
        elif isinstance(curr_node, ast.For):
            # Capture loop variables
            if isinstance(curr_node.target, ast.Name):
                variables.add(curr_node.target.id)
            elif isinstance(curr_node.target, ast.Tuple):
                for elt in curr_node.target.elts:
                    if isinstance(elt, ast.Name):
                        variables.add(elt.id)
        elif isinstance(curr_node, ast.Assign):
            # Handle tuple unpacking
            for target in curr_node.targets:
                if isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            variables.add(elt.id)

    for node in ast.walk(tree):
        visit_node(node)

    return sorted(list(variables))

# Example usage
demo_script = """
x = 10
y = 20
z = x + y
for i in range(10):
    print(i)
a, b = (1, 2)
def foo(c):
    d = c + 1
    return d
"""

demo_variables = ast_extract_variables(demo_script)
print(demo_variables)  # Output: ['a', 'b', 'c', 'd', 'i', 'x', 'y', 'z']