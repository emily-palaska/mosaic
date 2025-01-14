from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt

def initialize_model(model_name='sentence-transformers/all-MiniLM-L6-v2'):
    return SentenceTransformer(model_name)

def encode_labels(model, labels):
    return model.encode(labels)

def compute_similarity(embeddings):
    return util.pytorch_cos_sim(embeddings, embeddings)

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

def process_similarity(model, labels, plot=True, save_path='../plots/similarity_matrix.png'):
    embeddings, embedding_dim = encode_labels(model, labels)
    similarity_matrix = compute_similarity(embeddings)

    if plot:
        plot_similarity_matrix(similarity_matrix, save_path)

    return embeddings, similarity_matrix, embedding_dim
