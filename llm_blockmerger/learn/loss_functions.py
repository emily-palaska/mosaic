import torch

def vector_variance(vector_batch):
    assert vector_batch.shape[0] != 1, "Expected batch of vectors, not singular vector"
    mean = torch.mean(vector_batch, dim=0)
    squared_diffs = (vector_batch - mean) ** 2
    var_per_dim = torch.mean(squared_diffs, dim=0)
    return torch.mean(var_per_dim)

def pairwise_norm_cos_sim(batch1, batch2):
    a = batch1.unsqueeze(1)
    b = batch2.unsqueeze(0)

    dot_product = torch.sum(a * b, dim=-1)
    norm_a = torch.norm(a, dim=-1)
    norm_b = torch.norm(b, dim=-1)
    denominator = norm_a * norm_b + 1e-8

    cosine_sim = dot_product / denominator
    return (cosine_sim + 1) / 2

def transitive_contrastive_loss(a, b, c, a_out, c_out, threshold=0.98, alpha=1000):
    ab = pairwise_norm_cos_sim(a, b)
    bc = pairwise_norm_cos_sim(b, c)
    ac_out = pairwise_norm_cos_sim(a_out, c_out)

    labels = (ab * bc > threshold ** 2).float()

    similar = (1 - labels) * ac_out
    dissimilar = labels * (1 - ac_out)

    loss = alpha * torch.var(c_out) - similar - dissimilar
    return loss.mean()

def transitive_cross_entropy_loss(a, b, c, threshold=0.99, alpha=1.0):
    ab = pairwise_norm_cos_sim(a, b)
    bc = pairwise_norm_cos_sim(b, c)
    ac = pairwise_norm_cos_sim(a, c)
    assert torch.min(ac) >=0 and torch.max(ac) <= 1

    labels = (ab * bc > threshold ** 2).float()
    print(f'Percentage of 1s in the labels: {100 * labels.mean().item()}%')

    var_c = vector_variance(c)
    similar = labels * torch.log(ac + 1.0e-12)
    dissimilar = (1 - labels) * torch.log(1 - ac + 1.0e-12)

    loss = alpha * var_c - similar - dissimilar
    print(alpha*var_c.item(), similar.mean().item(), dissimilar.mean().item())
    return loss.mean()