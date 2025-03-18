import pytest
from llm_blockmerger.load.variable_extraction import extract_variables, Llama


@pytest.fixture
def script_data():
    scripts = [
"""
x = 10
y = 20
z = x + y
""",

"""
n = 10
for i in range(n):
    for j in range(n):
        if i == 0 and j == 0:
            continue
        else:
            print(i + j % 10)
""",

"""
a, b = (1, 2)
""",

"""
def foo(c):
    d = c + 1
    return d
""",

"""
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
    ]

    variable_sets = [
        ['x', 'y', 'z'],
        ['i', 'j', 'n'],
        ['a', 'b'],
        ['c', 'd'],
        ['a', 'b', 'c', 'd', 'i', 'x', 'y', 'z']
    ]

    return scripts, variable_sets

def test_ast_variable_extracting(script_data):
    scripts, variable_sets = script_data
    for script, variable_set in zip(scripts, variable_sets):
        assert extract_variables(script) == variable_set

def test_llama_variable_extracting(script_data):
    scripts, variable_sets = script_data
    llama = Llama()
    for script, variable_set in zip(scripts, variable_sets):
        assert extract_variables(script,model=llama) == variable_set