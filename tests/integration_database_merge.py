import os
os.chdir("../")

from llm_blockmerger.load import init_managers, flat_labels, create_blockdata, nb_variables
from llm_blockmerger.merge import embedding_synthesis, string_synthesis
from llm_blockmerger.store import BlockStore, HNSWVectorDB, InMemoryExactNNVectorDB
from llm_blockmerger.core import (
    plot_sim,
    pairwise_norm_cos_sim,
    LLM, print_merge_result,
    print_managers)

def preprocessing_pipeline(paths, verbose=True):
    managers = init_managers(paths)
    if verbose: print('Initialized managers...')
    print_managers(managers)

    #llama = LLM(task='question')
    #if verbose: print('Loaded llama model...')
    #for i, manager in enumerate(managers):
    #    print(f'\r{100 * i / len(managers) : .2f}%', end='')
    #    extract_notebook_variables(manager, llama)
    #if verbose: print('\rExtracted notebook variables...')

    embedding_model = LLM(task='embedding')
    if verbose: print('Initialized embedding model...')
    embeddings = embedding_model.encode(flat_labels(managers, code=True))
    if verbose: print(f'Encoded embeddings with shape {embeddings.shape}...')
    plot_sim(pairwise_norm_cos_sim(embeddings), './plots/similarity_matrix.png')
    if verbose: print('Plotted similarity matrix...')
    exit(0)

    vector_db = BlockStore(databasetype=HNSWVectorDB, empty=True)
    if verbose: print('Initialized vector database...')
    vector_db.create(embeddings=embeddings,
                     blockdata=create_blockdata(managers, embeddings))
    assert vector_db.num_docs() == len(embeddings), f'{vector_db.num_docs()} != {len(embeddings)}'
    if verbose: print(f'Loaded data to vector database with {vector_db.num_docs()} entries and {len(vector_db)} triplets...')
    return embedding_model, vector_db

def ready_database_pipeline(verbose=True):
    embedding_model = LLM(task='embedding')
    if verbose: print('Initialized embedding model...')

    vector_db = BlockStore(databasetype=HNSWVectorDB, empty=False)
    if verbose: print('Initialized vector database...')
    assert vector_db.num_docs() != 0, f'Empty BlockMergerVectorDB'
    return embedding_model, vector_db

def main():
    #paths = ['notebooks/example_more.ipynb', 'notebooks/pygrank_snippets.ipynb']
    paths = ['notebooks/02.02-The-Basics-Of-NumPy-Arrays.ipynb']
    specification = 'Initialize a logistic regression model. Use standardization on training inputs. Train the model.'

    embedding_model, vector_db = preprocessing_pipeline(paths)
    #embedding_model, vector_db = ready_database_pipeline()
    print_merge_result(specification,
                       string_synthesis(model=embedding_model, db=vector_db, spec=specification, var=False), title='STRING')
    print_merge_result(specification,
                       embedding_synthesis(model=embedding_model, db=vector_db, spec=specification, var=False), title='EMBEDDING')

if __name__ == '__main__':
    main()