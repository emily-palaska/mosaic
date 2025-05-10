import torch

def normalized_cosine_similarity(a, b):
    #cos_sim = F.cosine_similarity(embedding1, embedding2, dim=1)
    #return (cos_sim + 1) / 2 - 1.0e-6
    dot_product = torch.sum(a * b, dim=-1)
    norm_a = torch.norm(a, dim=-1)
    norm_b = torch.norm(b, dim=-1)
    denominator = norm_a * norm_b + 1e-8
    cosine_sim = dot_product / denominator
    return (cosine_sim + 1) / 2

def transitive_contrastive_loss(a, b, c, a_out, c_out, threshold=0.98, alpha=1000):
    ab = normalized_cosine_similarity(a, b)
    bc = normalized_cosine_similarity(b, c)
    ac_out = normalized_cosine_similarity(a_out, c_out)

    labels = (ab * bc > threshold ** 2).float()

    similar = (1 - labels) * ac_out
    dissimilar = labels * (1 - ac_out)

    loss = alpha * torch.var(c_out) - similar - dissimilar
    return loss.mean()

def transitive_cross_entropy_loss(a, b, c, a_out, c_out, threshold=0.98, alpha=1000):
    ab = normalized_cosine_similarity(a, b)
    bc = normalized_cosine_similarity(b, c)
    ac_out = normalized_cosine_similarity(a_out, c_out)

    labels = (ab * bc > threshold ** 2).float()

    similar = labels * torch.log(ac_out + 1.0e-12)
    dissimilar = (1 - labels) * torch.log(1 - ac_out + 1.0e-12)

    loss = alpha * torch.var(c_out) - similar - dissimilar
    return loss.mean()