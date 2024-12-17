from loading.blockloading import *
from encoding.embeddingmodel import *


def main():
    # Load a notebook
    notebook_paths = ['notebooks/02.02-The-Basics-Of-NumPy-Arrays.ipynb']
    blocks, labels = preprocess_blocks(*separate_blocks(load_notebooks(notebook_paths)))

    """    
    for block, label in zip(blocks, labels):
        print(label)
        print('CODE:')
        print(block)
        print('-'*40)
    """

    print('BLOCKS: ', len(blocks))
    print('LABELS: ', len(labels))

    # Embed and visualize labels
    embeddings, embedding_dim = encode_labels(initialize_model(), labels)
    plot_similarity_matrix(compute_similarity(embeddings), '../plots/similarity_matrix.png')
    print('EMBEDDINGS:', len(embeddings))

    exit(0)
"""
    # Initialize embedding db
    embedding_db = BlockMergerEmbeddingDB(workspace='./encoding')
    embedding_db.create(block_loader.labels)

    # Example query
    example = 'numpy'
    matches = embedding_db.read(example)

    print(f'MATCHES to {example}:')
    for m in matches:
        print('-' * 40)
        print(m.text)
"""
if __name__ == '__main__':
    main()