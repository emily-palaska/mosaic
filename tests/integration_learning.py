import os
os.chdir("../")

import torch.optim as optim
from torch.utils.data import DataLoader
from llm_blockmerger.store.vectordb import BlockMergerVectorDB, HNSWVectorDB
from llm_blockmerger.learn.mlp import MLP, train

def main():
    vector_db = BlockMergerVectorDB(databasetype=HNSWVectorDB, empty=False)
    print(f'Initialized vector database with {vector_db.get_num_docs()} entries and {len(vector_db)} triplets...')

    model = MLP(input_dim=vector_db.feature_size, layer_dims=[64, 32, 3])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print('Initialized MLP model...')

    train_loader = DataLoader(vector_db, batch_size=32, shuffle=False)
    print('Created train loader...')

    for batch in train_loader: pass
    print('Successful train loader pass...')

    train(model, train_loader, optimizer, epochs=10)
    print('Finished training...')

if __name__ == '__main__':
    main()