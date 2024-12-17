from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt

def initialize_model(model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Initialize the SentenceTransformer model.
    :param model_name: Name of the pre-trained model.
    :return: Initialized model.
    """
    return SentenceTransformer(model_name)

def encode_labels(model, labels):
    """
    Encode a list of sentences into embeddings.
    :param model: Initialized SentenceTransformer model.
    :param labels: List or numpy array of labels to encode.
    :return: Tuple of embeddings and embedding dimensions.
    """
    return model.encode(labels)

def compute_similarity(embeddings):
    """
    Compute the similarity matrix for embeddings.
    :param embeddings: Encoded sentence embeddings.
    :return: Similarity matrix.
    """
    return util.pytorch_cos_sim(embeddings, embeddings)

def plot_similarity_matrix(similarity_matrix, save_path='../plots/similarity_matrix.png'):
    """
    Plot and save the similarity matrix as an image.
    :param similarity_matrix: The similarity matrix to plot.
    :param save_path: Path to save the plot.
    """
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
    """
    Encode labels, compute similarity matrix, and optionally plot.
    :param model: Initialized SentenceTransformer model.
    :param labels: List of labels to encode.
    :param plot: Whether to plot the similarity matrix.
    :param save_path: Path to save the plot.
    :return: Tuple of embeddings, similarity matrix, and embedding dimensions.
    """
    embeddings, embedding_dim = encode_labels(model, labels)
    similarity_matrix = compute_similarity(embeddings)

    if plot:
        plot_similarity_matrix(similarity_matrix, save_path)

    return embeddings, similarity_matrix, embedding_dim
