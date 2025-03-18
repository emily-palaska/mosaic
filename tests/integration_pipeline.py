import os
os.chdir("../")

from llm_blockmerger.core.utils import *
from llm_blockmerger.core.models import LLM
from llm_blockmerger.store.vector_db import VectorDB, HNSWVectorDB
from llm_blockmerger.block_merge import linear_embedding_merge, linear_string_merge
from llm_blockmerger.load.variable_extraction import extract_notebook_variables

def main():
    notebook_paths = ['notebooks/02.02-The-Basics-Of-NumPy-Arrays.ipynb']
    managers = initialize_managers(notebook_paths)

    # Extract variables and descriptions with a LLAMA model
    model_name = "meta-llama/Llama-3.2-3B"
    llama = LLM(task='question', model_name=model_name, verbose=False)
    for manager in managers: extract_notebook_variables(manager, llama)

    blocks, labels, variables, var_descriptions, sources = concatenate_managers(managers)
    assert len(labels) == len(blocks), "Blocks and labels should have the same length"
    assert len(variables) == len(blocks), "Variables and blocks should have the same length"
    assert len(sources) == len(blocks), "Paths and blocks should have the same length"
    assert len(var_descriptions) == len(blocks), "Descriptions and blocks should have the same length"

    embedding_model = LLM(task='embedding')
    embeddings = embedding_model.encode_labels(labels)
    plot_similarity_matrix(compute_embedding_similarity(embeddings), './plots/similarity_matrix.png')
    print('EMBEDDINGS:', embeddings.shape)

    vector_db = VectorDB(dbtype=HNSWVectorDB, empty=True)
    assert vector_db.get_size() == 0, 'VectorDB should initialize empty'
    vector_db.create(labels, blocks, variables, var_descriptions, sources, embeddings)
    assert vector_db.get_size() == len(blocks), 'VectorDB should have created every block as a vector'

    example = 'numpy'
    print('MATCHES: ', len(vector_db.read(embedding_model.encode_labels(example))))

    specification = 'simple numpy program'
    print('LINEAR STRING MERGE')
    print_merge_result(specification, linear_string_merge(embedding_model, vector_db, specification))
    print('LINEAR EMBEDDING MERGE')
    print_merge_result(specification, linear_embedding_merge(embedding_model, vector_db, specification))


if __name__ == '__main__':
    main()