import os
os.chdir("../")

import torch.optim as optim
from torch.utils.data import DataLoader
from llm_blockmerger.store.vectordb import BlockMergerVectorDB, HNSWVectorDB
from llm_blockmerger.learn.mlp import MLP, train, triplet_cross_entropy_loss, transitive_contrastive_loss

def main():
    samples = 1000
    loss_function = triplet_cross_entropy_loss
    layer_dims, lr, batch_size, epochs = [128, 64, 32], 0.001, 1000, 10

    vector_db = BlockMergerVectorDB(databasetype=HNSWVectorDB, empty=False, training_samples=samples)
    print(f'Initialized vector database with {vector_db.get_num_docs()} entries and {len(vector_db)} training samples...')

    model = MLP(input_dim=vector_db.feature_size, layer_dims=layer_dims, loss_function=loss_function)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print('Initialized MLP model...')

    train_loader = DataLoader(vector_db, batch_size=batch_size, shuffle=False)
    print('Created train loader...')

    train(model, train_loader, optimizer, epochs=epochs)
    print('Finished training...')

if __name__ == '__main__':
    main()