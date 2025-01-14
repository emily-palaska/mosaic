from llm_blockmerger.loading.blockloading import *
from llm_blockmerger.encoding.vector_db import *
from llm_blockmerger.encoding.embedding_model import *
from llm_blockmerger.linear_merging import *

def main():
    # Load a notebook
    notebook_paths = ['notebooks/02.02-The-Basics-Of-NumPy-Arrays.ipynb']
    blocks, labels = preprocess_blocks(*separate_blocks(load_notebooks(notebook_paths)))
    assert len(blocks) == len(labels), "Blocks and labels should have the same length"

    """
    # Prints the extracted blocks
    for block, l abel in zip(blocks, labels):
        print(label)
        print('CODE:')
        print(block)
        print('-'*40)
    """

    # Embed and visualize labels
    embedding_model = initialize_model()
    embeddings = encode_labels(embedding_model, labels)
    #plot_similarity_matrix(compute_similarity(embeddings), './plots/similarity_matrix.png')
    print('EMBEDDINGS:', embeddings.shape)

    # Create a vector database, and try it with an add/read example
    # TODO test assertation of shape with add/read example
    example = 'numpy'
    vector_db = (VectorDB(dbtype=HNSWVectorDB))
    vector_db.create(labels, embeddings, blocks)
    print('MATCHES: ', len(vector_db.read(encode_labels(embedding_model, [example]))))

    """
    # Prints the labels and code blocks of the matches
    for m in matches:
        print('-' * 40)
        print(m.label)
        print(m.block)
        print('-'*40)
    """

    # Code generation pipeline through a linear search
    example = 'simple numpy program'
    # Choice of merging method: string or embedding comparison
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