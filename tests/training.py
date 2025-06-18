import os
os.chdir("../")

from torch.optim import Adam
from llm_blockmerger.store import BlockDB
from llm_blockmerger.learn import MLP, train, TransitiveCrossEntropy, TransitiveContrastive, train_plot, loaders


def integration(plot=True, verbose=True):
    loss_func = TransitiveCrossEntropy(var=False, mean=False)
    layer_dims, lr, batch, epochs = [32, 32, 32], 0.001, 512, 300

    db = BlockDB(empty=False)
    model = MLP(input_dim=db.features, layer_dims=layer_dims)
    optimizer = Adam(model.parameters(), lr=lr)

    train_loader, val_loader = loaders(db, batch, train_split=0.8)
    results = train(model, train_loader, val_loader, optimizer, loss_func, epochs, verbose=verbose)
    if plot: train_plot(results, path='./plots/')

def validation():
    """
    after training, the whole dataset passes through the model
    load the db can keep the old embeddings
    assign new embeddings and create dataset
    do some query merging
    before closing restore the original embeddings
    """
    return NotImplemented

if __name__ == '__main__':
    integration()