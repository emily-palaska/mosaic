from loading.blockloading import *
from encoding.embeddingdb import BlockMergerEmbeddingDB


def main():
    # Initialize block loader which will keep the code with the respective labels
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

    """
    for i, label in enumerate(block_loader.labels):
        print(label)
        #print(block_loader.blocks[i])
        print('-' * 40)
        
    # Initialize the embedding model
    embedding_model = EmbeddingModel()
    embedding_model.encode(block_loader.labels)
    embedding_model.similarity()
    print('EMBEDDINGS: ', embedding_model.embeddings.shape)
    """
    exit(0)

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

if __name__ == '__main__':
    main()