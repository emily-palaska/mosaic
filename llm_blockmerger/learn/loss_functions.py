import torch.nn.functional as F
import torch

def normalized_cosine_similarity(embedding1, embedding2):
    cos_sim = F.cosine_similarity(embedding1, embedding2, dim=1)
    return (cos_sim + 1) / 2

def transitive_contrastive_loss(a, b, c, threshold=0.8):
    # Normalized cosine distances
    ab = normalized_cosine_similarity(a, b)
    bc = normalized_cosine_similarity(b, c)
    ca = normalized_cosine_similarity(c, a)

    labels = (ab * bc > threshold ** 2).float()
    loss_similar = (1 - labels) * ca
    loss_dissimilar = labels * (1 - ca)

    return torch.mean(0.5 * (loss_similar + loss_dissimilar))

def triplet_cross_entropy_loss(a, b, c, threshold=0.8):
    ab = normalized_cosine_similarity(a, b)
    bc = normalized_cosine_similarity(b, c)
    ac = normalized_cosine_similarity(a, c)

    labels = (ab * bc > threshold ** 2).float()
    loss = - (labels * torch.log(ac) + (1 - labels) * torch.log(1 - ac))

    return loss.mean()