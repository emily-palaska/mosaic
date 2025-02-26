import os
os.chdir("../")

from llm_blockmerger.loading.blockloading import concatenate_managers, preprocess_blocks, load_notebooks
from llm_blockmerger.encoding.vector_db import VectorDB, HNSWVectorDB
from llm_blockmerger.encoding.embedding_model import initialize_model, encode_labels
from llm_blockmerger.linear_merging import linear_embedding_merge, linear_string_merge

def main():
    # Load a notebook
    notebook_paths = ['notebooks/02.02-The-Basics-Of-NumPy-Arrays.ipynb']
    blocks, labels, variables = concatenate_managers(preprocess_blocks(load_notebooks(notebook_paths)))
    assert len(blocks) == len(labels), "Blocks and labels should have the same length"
    assert len(variables) == len(notebook_paths), "Variables and paths should have the same length"

    # Embed and visualize labels
    embedding_model = initialize_model()
    embeddings = encode_labels(embedding_model, labels)
    #plot_similarity_matrix(compute_similarity(embeddings), './plots/similarity_matrix.png')
    print('EMBEDDINGS:', embeddings.shape)

    # Create a vector database
    vector_db = VectorDB(dbtype=HNSWVectorDB, empty=True)
    assert vector_db.get_size() == 0, 'VectorDB should initialize empty'
    vector_db.create(labels, embeddings, blocks)
    assert vector_db.get_size() == len(blocks), 'VectorDB should have created every block as a vector'

    # Read from the database
    example = 'numpy'
    print('MATCHES: ', len(vector_db.read(encode_labels(embedding_model, [example]))))

    # Code generation pipeline through a linear string or embedding search
    example = 'simple numpy program'
    #labels, blocks = linear_string_merge(embedding_model, vector_db, example)
    labels, blocks = linear_embedding_merge(embedding_model, vector_db, example)
    print('MERGE RESULT:')

    # Print the labels and code blocks of the result
    for label, block in zip(labels, blocks):
        print('-' * 40)
        print(label)
        print('CODE:')
        print(block)


if __name__ == '__main__':
    main()