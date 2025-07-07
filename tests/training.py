import os
os.chdir("../")


from torch.optim import Adam
from mosaic.store import BlockDB
from mosaic.learn import MLP, train, TransitiveCrossEntropy, TransitiveContrastive, train_plot, loaders
from tests.core import merge, deploy_mlp

def integration(plot=True, verbose=True, save=True):
    loss_func = TransitiveCrossEntropy()
    layer_dims, lr, batch, epochs = [32, 32, 32], 0.001, 512, 60

    db = BlockDB(empty=False)
    model = MLP(input_dim=db.features, layer_dims=layer_dims)
    optimizer = Adam(model.parameters(), lr=lr)

    train_loader, val_loader = loaders(db, batch, train_split=0.8)
    results = train(model, train_loader, val_loader, optimizer, loss_func, epochs, verbose=verbose)
    if plot: train_plot(results, path='./plots/')
    if save: model.save(path='./results/models/ce_default')

def validation():
    db = BlockDB(empty=False)
    embeddings, blockdata, features = db.embeddings(), db.blockdata(), db.features

    model = MLP(input_dim=db.features).load(path='./results/models/ce_default', device='cpu')
    assert model.device == 'cpu', 'Model initialized in gpu'
    deploy_mlp(model, embeddings, blockdata)

    demo = ['Initialize a logistic regression model. Use standardization on training inputs. Train the model.']
    merge(demo, save=False, mlp=model)

    db = BlockDB(features=features,empty=True)
    db.create(embeddings, blockdata)
    print('Restored old embeddings')

if __name__ == '__main__':
    integration()