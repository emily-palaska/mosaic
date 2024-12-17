from encoding.vector_db import vectordb_read
from encoding.embedding_model import encode_labels
def linear_merge(embedding_model, vector_db, specification, max_iterations=10):
    """
    Iteratively finds the nearest neighbor to a given specification,
    retains the label and block, removes common words from the label
    and specification, and repeats until the specification is empty
    or the maximum number of iterations is reached.

    :param embedding_model: Pre-trained sentence embedding model.
    :param vector_db: Initialized vector database for storing and querying embeddings.
    :param specification: String specification to process iteratively.
    :param max_iterations: Maximum number of iterations to perform (default is 10).
    :return: List of tuples containing (label, block) for each iteration.
    """
    results = []  # Store the (label, block) results from each iteration

    for _ in range(max_iterations):
        # Step 1: Encode the current specification to an embedding
        spec_embedding = encode_labels(embedding_model, [specification])

        # Step 2: Query the vector database for the nearest neighbor
        nearest_neighbors = vectordb_read(vector_db, spec_embedding, limit=1)
        if not nearest_neighbors:
            break  # Exit if no neighbors are found

        # Step 3: Extract and store the (label, block) pair of the nearest neighbor
        nearest_doc = nearest_neighbors[0]
        label = nearest_doc.label
        block = nearest_doc.block
        results.append((label, block))

        # Step 4: Remove common words from the label and specification, update the specification
        label_words = set(label.split())
        spec_words = set(specification.split())
        remaining_words = spec_words - label_words
        if not remaining_words:
            break  # Exit if no words are left in the specification
        specification = " ".join(remaining_words)

    return results
