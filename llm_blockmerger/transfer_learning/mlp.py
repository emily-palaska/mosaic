import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, input_dim = 128, layer_dims = None):
        self.input_dim = input_dim
        self.layer_dims = layer_dims if layer_dims else [64, 32]
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in layer_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def loss_function(self, *args, **kwargs):
        raise NotImplementedError("Loss function not implemented")


# Contrastive Loss Function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, label):
        distance = torch.norm(embedding1 - embedding2, p=2, dim=1)

        loss = (label * torch.pow(distance, 2) +  # For similar pairs (label = 1)
                (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0),
                                        2))  # For dissimilar pairs (label = 0)

        return loss.mean()

def main():
    input_dim = 5
    layer_dims = [64, 32]
    batch_size = 10

    model = MLP(input_dim, layer_dims)

    embeddings1 = torch.tensor([1, 1, 1, 1, 1])
    embeddings2 = torch.tensor([1, 1, 1, 1, 1])

    labels = torch.randint(0, 2, (batch_size,))

    output1 = model(embeddings1)
    output2 = model(embeddings2)

    contrastive_loss = ContrastiveLoss(margin=1.0)
    loss = contrastive_loss(output1, output2, labels)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Contrastive Loss: {loss.item()}")

if __name__ == "__main__":
    main()
