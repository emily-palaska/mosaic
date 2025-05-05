import torch.nn.functional as F

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

    #print(f'\t Label distribution: {torch.sum(label, dim=0) / label.shape[0]}')
    #print(f'\t Similar: {torch.mean(loss_similar)} Dissimilar: {torch.mean(loss_dissimilar)}')
    return torch.mean(0.5 * (loss_similar + loss_dissimilar))

def triplet_cross_entropy_loss(a, b, c, threshold=0.8):
    ab = F.cosine_similarity(a, b, dim=-1)
    bc = F.cosine_similarity(b, c, dim=-1)
    ac = F.cosine_similarity(a, c, dim=-1)

    labels = (ab * bc > threshold ** 2).float()
    return F.cross_entropy(ac, labels)