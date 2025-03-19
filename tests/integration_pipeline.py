import os
os.chdir("../")

from llm_blockmerger.core.utils import *
from llm_blockmerger.core.models import LLM
from llm_blockmerger.store.vector_db import VectorDB, HNSWVectorDB
from llm_blockmerger.block_merge import linear_embedding_merge, linear_string_merge
from llm_blockmerger.load.variable_extraction import extract_notebook_variables
from llm_blockmerger.variable_merge import merge_variables

def main():
    notebook_paths = ['notebooks/02.02-The-Basics-Of-NumPy-Arrays.ipynb']
    managers = initialize_managers(notebook_paths)
    print('Initialized managers...')

    # Extract variables and descriptions with a LLAMA model
    llama = LLM(task='question')
    print('Loaded llama model...')
    for manager in managers: extract_notebook_variables(manager, llama)
    print('Extracted notebook variables...')

    blocks, labels, variables, var_descriptions, sources = concatenate_managers(managers)
    assert len(labels) == len(blocks), "Blocks and labels should have the same length"
    assert len(sources) == len(blocks), "Paths and blocks should have the same length"
    assert len(variables) == len(blocks), "Variables and blocks should have the same length"
    assert len(var_descriptions) == len(blocks), "Descriptions and blocks should have the same length"
    print('Passed dimension assertations...')

    embedding_model = LLM(task='embedding')
    print('Initialized embedding model...')
    embeddings = embedding_model.encode_strings(labels)
    print(f'Encoded embeddings with shape {embeddings.shape}...')
    plot_similarity_matrix(compute_embedding_similarity(embeddings), './plots/similarity_matrix.png')
    print('Plotted similarity matrix...')

    vector_db = VectorDB(dbtype=HNSWVectorDB, empty=True)
    print('Initialized vector database...')
    assert vector_db.get_size() == 0, 'VectorDB should initialize empty'
    vector_db.create(labels, blocks, variables, var_descriptions, sources, embeddings)
    assert vector_db.get_size() == len(blocks), 'VectorDB should have created every block as a vector'
    print('Loaded data to vector database...')

    specification = 'simple numpy program'
    print_merge_result(specification, merge_variables(*linear_string_merge(embedding_model, vector_db, specification)))
    print_merge_result(specification, merge_variables(*linear_embedding_merge(embedding_model, vector_db, specification)))


if __name__ == '__main__':
    main()