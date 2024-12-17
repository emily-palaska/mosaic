from loading.blockloading import *
from encoding.embeddingmodel import *
from encoding.embeddingdb import *

def main():
    # Load a notebook
    notebook_paths = ['notebooks/02.02-The-Basics-Of-NumPy-Arrays.ipynb']
    blocks, labels = preprocess_blocks(*separate_blocks(load_notebooks(notebook_paths)))
    print('BLOCKS: ', len(blocks))
    print('LABELS: ', len(labels))
    """    
    for block, label in zip(blocks, labels):
        print(label)
        print('CODE:')
        print(block)
        print('-'*40)
    """

    # Embed and visualize labels
    embedding_model = initialize_model()
    embeddings = encode_labels(embedding_model, labels)
    plot_similarity_matrix(compute_similarity(embeddings), '../plots/similarity_matrix.png')
    print('EMBEDDINGS:', embeddings.shape)

    # Create a vector database, and try it with an add/read example
    db = initialize_vectordb()
    vectordb_create(db, labels, embeddings)
    print(encode_labels(embedding_model, ['numpy']).shape)
    matches = vectordb_read(db, encode_labels(embedding_model, ['numpy']))
    print('MATCHES: ', len(matches))
    """
    for m in matches:
        print('-' * 40)
        print(m.text)
    """
if __name__ == '__main__':
    main()