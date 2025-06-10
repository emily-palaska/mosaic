import os
os.chdir("../")

import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from llm_blockmerger.store import BlockMergerVectorDB, HNSWVectorDB
from llm_blockmerger.learn import MLP, train, TransitiveCrossEntropyLoss, visualize_results

def main():
    loss_function = TransitiveCrossEntropyLoss()
    layer_dims, lr, batch_size, epochs = [128, 64, 32], 0.001, 100, 120

    vector_db = BlockMergerVectorDB(databasetype=HNSWVectorDB, empty=False)
    print(f'Initialized vector database with {vector_db.get_num_docs()} entries and {len(vector_db)} training samples...')

    model = MLP(input_dim=vector_db.feature_size, layer_dims=layer_dims)
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