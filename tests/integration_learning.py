import os
os.chdir("../")

import torch.optim as optim
from llm_blockmerger.store.vector_db import VectorDB, HNSWVectorDB
from llm_blockmerger.learn.mlp import MLP, train
from torch.utils.data import DataLoader

def main():
    vector_db = VectorDB(dbtype=HNSWVectorDB, empty=False)
    assert len(vector_db) == 84, 'VectorDB should initialize with saved elements'
    print('Initialized vector database...')

    model = MLP(input_dim=vector_db.get_feature_size(), layer_dims=[64, 32, 3])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print('Initialized model...')

    train_loader = DataLoader(vector_db, batch_size=32, shuffle=True)
    print('Created train loader...')

    train(model, train_loader, optimizer, epochs=10)

if __name__ == "__main__":
    main()