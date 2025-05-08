import os
os.chdir("../")

from llm_blockmerger.load import initialize_managers, extract_labels, create_blockdata, extract_notebook_variables
from llm_blockmerger.merge import merge_variables, linear_embedding_merge, linear_string_merge
from llm_blockmerger.store import BlockMergerVectorDB, HNSWVectorDB, InMemoryExactNNVectorDB
from llm_blockmerger.core import (
    plot_similarity_matrix,
    compute_embedding_similarity,
    LLM, print_merge_result,
    print_managers)

def preprocessing_pipeline(paths, verbose=True):
    managers = initialize_managers(paths)
    if verbose: print('Initialized managers...')
    print_managers(managers)

    llama = LLM(task='question')
    if verbose: print('Loaded llama model...')
    for i, manager in enumerate(managers):
        print(f'\r{100 * i / len(managers) : .2f}%', end='')
        extract_notebook_variables(manager, llama)
    if verbose: print('\rExtracted notebook variables...')

    embedding_model = LLM(task='embedding')
    if verbose: print('Initialized embedding model...')
    embeddings = embedding_model.encode_strings(extract_labels(managers, blocks=True))
    if verbose: print(f'Encoded embeddings with shape {embeddings.shape}...')
    plot_similarity_matrix(compute_embedding_similarity(embeddings), './plots/similarity_matrix.png')
    if verbose: print('Plotted similarity matrix...')

    for manager in managers: merge_variables(embedding_model, manager)
    if verbose: print('Merged variables...')

    vector_db = BlockMergerVectorDB(databasetype=HNSWVectorDB, empty=True)
    if verbose: print('Initialized vector database...')
    vector_db.create(embeddings=embeddings,
                     blockdata=create_blockdata(managers, embeddings))
    assert vector_db.get_num_docs() == len(embeddings), f'{vector_db.get_num_docs()} != {len(embeddings)}'
    if verbose: print(f'Loaded data to vector database with {vector_db.get_num_docs()} entries and {len(vector_db)} triplets...')
    return embedding_model, vector_db

def ready_database_pipeline(verbose=True):
    embedding_model = LLM(task='embedding')
    if verbose: print('Initialized embedding model...')

    vector_db = BlockMergerVectorDB(databasetype=HNSWVectorDB, empty=False)
    if verbose: print('Initialized vector database...')
    assert vector_db.get_num_docs() != 0, f'Empty BlockMergerVectorDB'
    return embedding_model, vector_db

def main():
    paths = ['notebooks/example.py', 'notebooks/example_more.py', 'notebooks/pygrank_snippets.py']
    #paths = ['notebooks/02.02-The-Basics-Of-NumPy-Arrays.ipynb']
    specification = 'a simple program that trains a model'

    embedding_model, vector_db = preprocessing_pipeline(paths)
    #embedding_model, vector_db = ready_database_pipeline()
    print_merge_result(specification,
                       linear_string_merge(embedding_model, vector_db, specification),
                       merge_type='STRING')
    print_merge_result(specification,
                       linear_embedding_merge(embedding_model, vector_db, specification),
                       merge_type='EMBEDDING')

if __name__ == '__main__':
    main()