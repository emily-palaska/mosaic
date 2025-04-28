import os

os.chdir("../")

from llm_blockmerger.core.embeddings import plot_similarity_matrix, compute_embedding_similarity
from llm_blockmerger.load.managers import initialize_managers, extract_labels, create_blockdata
from llm_blockmerger.core.models import LLM
from llm_blockmerger.load.variable_extraction import extract_notebook_variables
from llm_blockmerger.variable_merge import merge_variables
from llm_blockmerger.store.vector_db import VectorDB, HNSWVectorDB
from llm_blockmerger.core.utils import print_merge_result
from llm_blockmerger.block_merge import linear_embedding_merge, linear_string_merge

def main():
    notebook_paths = ['notebooks/02.02-The-Basics-Of-NumPy-Arrays.ipynb']
    managers = initialize_managers(notebook_paths)
    print('Initialized managers...')

    llama = LLM(task='question')
    print('Loaded llama model...')
    for manager in managers: extract_notebook_variables(manager, llama)
    print('Extracted notebook variables...')

    embedding_model = LLM(task='embedding')
    print('Initialized embedding model...')
    embeddings = embedding_model.encode_strings(extract_labels(managers))
    print(f'Encoded embeddings with shape {embeddings.shape}...')
    plot_similarity_matrix(compute_embedding_similarity(embeddings), './plots/similarity_matrix.png')
    print('Plotted similarity matrix...')

    for manager in managers: merge_variables(embedding_model, manager)
    print('Merged variables...')

    vector_db = VectorDB(databasetype=HNSWVectorDB, empty=True)
    print('Initialized vector database...')
    vector_db.create(embeddings=embeddings,
                     blockdata=create_blockdata(managers, embeddings))
    assert vector_db.get_num_docs() == len(embeddings), f'VectorDB length is {vector_db.get_num_docs()} instead of {len(embeddings)}'
    print(f'Loaded data to vector database with {vector_db.get_num_docs()} entries and {len(vector_db)} triplets...')
    exit(0)

    # Merge example
    #specification = 'simple numpy program'
    #print_merge_result(specification, merge_variables(*linear_string_merge(embedding_model, vector_db, specification)))
    #print_merge_result(specification, merge_variables(*linear_embedding_merge(embedding_model, vector_db, specification)))


if __name__ == '__main__':
    main()