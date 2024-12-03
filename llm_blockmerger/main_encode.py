from loading.blockloader import BlockLoader
from encoding.embeddingmodel import EmbeddingModel

def main():
    notebook_paths = ['notebooks/02.02-The-Basics-Of-NumPy-Arrays.ipynb']
    block_loader = BlockLoader(notebook_paths)
    print('BLOCKS')
    print(len(block_loader.blocks))

    """
    for i, label in enumerate(block_loader.labels):
        print(label)
        #print(block_loader.blocks[i])
        print('-' * 40)
    """

    embedding_model = EmbeddingModel()
    embedding_model.encode(block_loader.labels)
    embedding_model.similarity()

    print('EMBEDDINGS')
    print(embedding_model.embeddings.shape)



if __name__ == '__main__':
    main()