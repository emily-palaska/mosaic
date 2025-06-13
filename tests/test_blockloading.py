import pytest
from llm_blockmerger.load.scripts import *

@pytest.fixture
def notebook_data():
    path = '../notebooks/test_notebook.ipynb'
    blocks = [
        ['    return a + b '],
        ['def add(a, b):\n', '    return a + b '],
        ['print(result)  '],
        ['result = add(2, 3)\n', 'print(result)  '],
        ['print(result) '],
        ['result = add(-1, 1)\n', 'print(result) ']
    ]

    labels = [
        'MARKDOWN: # Functions in python\nSimple addition function\nCOMMENT: This is a simple addition function',
        'MARKDOWN: # Functions in python\nSimple addition function\nCOMMENT: Example of usage:',
        'MARKDOWN: Testing it with two examples\nCOMMENT: Should print 5',
        'MARKDOWN: Testing it with two examples\nCOMMENT: ',
        'MARKDOWN: Testing it with two examples\nCOMMENT: Should print 0',
        'MARKDOWN: Testing it with two examples\nCOMMENT: '
    ]

    variables = [['a', 'b'], ['a', 'b'], ['result'], ['a', 'result'], ['result'], ['a', 'result']]
    return [path], blocks, labels, variables

def test_preprocess_pipeline(notebook_data):
    path, expected_blocks, expected_labels, expected_variables = notebook_data
    blocks, labels, variables = concatenate_managers(preprocess_blocks(load_notebooks(path)))
    assert len(blocks) == len(labels)
    assert blocks == expected_blocks
    assert labels == expected_labels
    assert variables == expected_variables