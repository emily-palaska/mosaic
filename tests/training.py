import os
os.chdir("../")

import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from llm_blockmerger.store import BlockStore, HNSWVectorDB
from llm_blockmerger.learn import MLP, train, TransitiveCrossEntropyLoss, visualize_results
from llm_blockmerger.core import pairwise_norm_cos_sim, plot_sim

def main():
    loss_function = TransitiveCrossEntropyLoss(mean=False, var=False)
    layer_dims, lr, batch_size, epochs = [32, 32, 32], 0.001, 128, 60

    vector_db = BlockStore(databasetype=HNSWVectorDB, empty=False)
    print(f'Initialized vector database with {vector_db.num_docs()} entries and {len(vector_db)} training samples...')
    plot_sim(pairwise_norm_cos_sim(vector_db.embeddings()), path='./plots/similarity_matrix.png')

    exit(0)
    model = MLP(input_dim=vector_db.features, layer_dims=layer_dims)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print('Initialized MLP model...')


    train_size = int(0.8 * len(vector_db))
    val_size = len(vector_db) - train_size
    train_dataset, val_dataset = random_split(vector_db, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    results = train(model, train_loader, val_loader, optimizer, loss_function=loss_function, epochs=epochs)
    visualize_results(results, path='./plots/')

if __name__ == '__main__':
    main()