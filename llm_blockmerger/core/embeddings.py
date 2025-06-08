import matplotlib.pyplot as plt
import torch

def embedding_projection(current_embedding, neighbor_embedding):
    if not isinstance(current_embedding, torch.Tensor): current_embedding = torch.tensor(current_embedding)
    if not isinstance(neighbor_embedding, torch.Tensor): neighbor_embedding = torch.tensor(neighbor_embedding)

    if torch.all(current_embedding == 0): return None
    inner_product = torch.dot(neighbor_embedding, current_embedding) / torch.dot(current_embedding, current_embedding)
    return inner_product * current_embedding

def pairwise_norm_cos_sim(batch1, batch2=None, dtype=torch.float):
    if not isinstance(batch1, torch.Tensor): batch1 = torch.tensor(batch1, dtype=dtype)
    if not isinstance(batch2, torch.Tensor): batch2 = torch.tensor(batch2, dtype=dtype)

    a = batch1.unsqueeze(1)
    b = batch2.unsqueeze(0) if batch2 is not None else batch1

    dot_product = torch.sum(a * b, dim=-1)
    norm_a = torch.norm(a, dim=-1)
    norm_b = torch.norm(b, dim=-1)
    denominator = norm_a * norm_b + 1e-8

    cosine_sim = dot_product / denominator
    return (cosine_sim + 1) / 2

def vector_variance(vector_batch):
    assert vector_batch.shape[0] != 1, "Expected batch of vectors, not singular vector"
    mean = torch.mean(vector_batch, dim=0)
    squared_diffs = (vector_batch - mean) ** 2
    var_per_dim = torch.mean(squared_diffs, dim=0)
    return torch.mean(var_per_dim)

def plot_similarity_matrix(similarity_matrix, save_path='../plots/similarity_matrix.png'):
    plt.figure(figsize=(12, 12))
    plt.imshow(similarity_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Similarity')

    # Annotate the matrix with similarity values - ONLY for small dimensions due to visibility
    if similarity_matrix.shape[0] <= 20:
        for i in range(similarity_matrix.shape[0]):
            for j in range(similarity_matrix.shape[1]):
                plt.text(j, i, f"{similarity_matrix[i, j]:.2f}",
                         ha="center", va="center", color="black")

    plt.title("Similarity Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()