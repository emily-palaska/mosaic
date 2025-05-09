import torch.nn.functional as F
from torch.nn import BCELoss
import torch

def normalized_cosine_similarity(embedding1, embedding2):
    cos_sim = F.cosine_similarity(embedding1, embedding2, dim=1)
    return (cos_sim + 1) / 2 - 1.0e-6

def transitive_contrastive_loss(a, b, c, c_out, threshold=0.8):
    ab = normalized_cosine_similarity(a, b)
    bc = normalized_cosine_similarity(b, c)
    ca = normalized_cosine_similarity(c, a)

    labels = (ab * bc > threshold ** 2).float()
    loss_similar = (1 - labels) * ca
    loss_dissimilar = labels * (1 - ca)

    return torch.mean(0.5 * (loss_similar + loss_dissimilar))

def triplet_cross_entropy_loss(a, b, c, a_out, c_out, threshold=0.98, alpha=1000):
    ab = normalized_cosine_similarity(a, b)
    bc = normalized_cosine_similarity(b, c)
    ac_out = normalized_cosine_similarity(a_out, c_out)

    labels = (ab * bc > threshold ** 2).float()

    similar = labels * torch.log(ac_out + 1.0e-12)
    dissimilar = (1 - labels) * torch.log(1 - ac_out + 1.0e-12)

    loss = alpha * torch.var(c_out) - similar - dissimilar
    return loss.mean()