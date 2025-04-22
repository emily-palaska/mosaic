import os
os.chdir("../")

import torch.optim as optim
from torch.utils.data import DataLoader
from llm_blockmerger.core.embeddings import plot_similarity_matrix, compute_embedding_similarity
from llm_blockmerger.load.managers import initialize_managers, concatenate_managers
from llm_blockmerger.core.models import LLM
from llm_blockmerger.load.variable_extraction import extract_notebook_variables
from llm_blockmerger.variable_merge import merge_variables
from llm_blockmerger.store.vector_db import VectorDB, HNSWVectorDB
from llm_blockmerger.learn.mlp import MLP, train
from llm_blockmerger.core.utils import print_merge_result
from llm_blockmerger.block_merge import linear_embedding_merge, linear_string_merge

def main():
    notebook_paths = ['notebooks/02.02-The-Basics-Of-NumPy-Arrays.ipynb']
    managers = initialize_managers(notebook_paths)
    print('Initialized managers...')

    # Extract variables and descriptions with a LLAMA model
    llama = LLM(task='question')
    print('Loaded llama model...')
    for manager in managers: extract_notebook_variables(manager, llama)
    print('Extracted notebook variables...')

    blocks, labels, variable_dictionaries, sources = concatenate_managers(managers)
    assert len(labels) == len(blocks), "Blocks and labels should have the same length"
    assert len(sources) == len(blocks), "Paths and blocks should have the same length"
    assert len(variable_dictionaries) == len(blocks), "Variable dictionaries and blocks should have the same length"
    print('Passed dimension assertations...')

    embedding_model = LLM(task='embedding')
    print('Initialized embedding model...')
    embeddings = embedding_model.encode_strings(labels)
    print(f'Encoded embeddings with shape {embeddings.shape}...')
    plot_similarity_matrix(compute_embedding_similarity(embeddings), './plots/similarity_matrix.png')
    print('Plotted similarity matrix...')

    for manager in managers: merge_variables(embedding_model, manager)
    print('Merged variables...')

    vector_db = VectorDB(databasetype=HNSWVectorDB, empty=True)
    print('Initialized vector database...')
    assert len(vector_db) == 0, 'VectorDB should initialize empty'
    vector_db.create(labels, blocks, variable_dictionaries, sources, embeddings)
    assert len(vector_db) == len(blocks), 'VectorDB should have created every block as a vector'
    print(f'Loaded data to vector database with size {len(vector_db)}...')

    model = MLP(input_dim=vector_db.get_feature_size(), layer_dims=[64, 32, 3])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print('Initialized model...')

    train_loader = DataLoader(vector_db, batch_size=32, shuffle=True)
    print('Created train loader...')

    train(model, train_loader, optimizer, epochs=10)
    print('Finished training...')

    # Merge example
    #specification = 'simple numpy program'
    #print_merge_result(specification, merge_variables(*linear_string_merge(embedding_model, vector_db, specification)))
    #print_merge_result(specification, merge_variables(*linear_embedding_merge(embedding_model, vector_db, specification)))


if __name__ == '__main__':
    main()