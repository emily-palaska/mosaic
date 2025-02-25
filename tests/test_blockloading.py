import pytest
from llm_blockmerger.loading.blockloading import *

@pytest.fixture
def notebook_data():
    path = './notebooks/test_notebook.ipynb'
    blocks = [
        ['def add(a, b):', 'return a + b'],
        ['print(result)'],
        ['result = add(2, 3)'],
        ['print(result)'],
        ['result = add(-1, 1)']
    ]

    labels = [
        'MARKDOWN: # Functions in python\nSimple addition function\nCOMMENT:  Example of usage:',
        'MARKDOWN: Testing it with two examples\nCOMMENT: Should print 5',
        'MARKDOWN: Testing it with two examples\nCOMMENT: ',
        'MARKDOWN: Testing it with two examples\nCOMMENT: Should print 0',
        'MARKDOWN: Testing it with two examples\nCOMMENT: '
    ]

    return [path], blocks, labels

def test_preprocess_pipeline(notebook_data):
    path, expected_blocks, expected_labels = notebook_data
    blocks, labels = concatenate_managers(preprocess_blocks(load_notebooks(path)))
    assert len(blocks) == len(labels)
    assert blocks == expected_blocks
    assert labels == expected_labels