import numpy as np
from llm_blockmerger.encoding.vector_db import vectordb_read
from llm_blockmerger.encoding.embedding_model import encode_labels


def remove_common_words(original: str, to_remove: str, replacement='UNKNOWN') -> str:
    """
    Replaces specific words in the original string with a word (default is 'UNKNOWN') in a case-insensitive manner,
    while preserving the order of the words. Also replaces all newline characters (`\n`)
    in the original string with spaces before processing.

    :param original: The original string where words will be replaced.
    :param to_remove: A space-separated string of words to replace with 'UNKNOWN' in the original string.
    :param replacement: The replacement string (default is 'UNKNOWN')
    :return: A string with the words from `to_remove` replaced with 'UNKNOWN', maintaining the order of the original words.

    Example: "a Delicious green apple\nred" - "green red blue" = "a Delicious UNKNOWN apple UNKNOWN"
    """
    # Replace newline characters with spaces
    original = original.replace('\n', ' ')

    # Split strings into lists of lowercase words
    original_words = original.split()
    remove_words = set(word.lower() for word in to_remove.split())

    # Replace common words with replacement in a case-insensitive manner
    replaced_words = [
        replacement if word.lower() in remove_words else word
        for word in original_words
    ]

    # Join the replaced words back into a string
    return ' '.join(replaced_words)

def linear_string_merge(embedding_model, vector_db, specification, max_iterations=10):
    """
    Iteratively finds the nearest neighbor to a given specification,
    retains the label and block, removes common words from the label
    and specification, replacing them with a neutral word and repeats
    until the specification is empty or the maximum number of
    iterations is reached.

    :param embedding_model: Pre-trained sentence embedding model.
    :param vector_db: Initialized vector database for storing and querying embeddings.
    :param specification: String specification to process iteratively.
    :param max_iterations: Maximum number of iterations to perform (default is 10).
    :return: List of tuples containing (label, block) for each iteration.
    """
    # Store the label and block results from each iteration separately
    labels = []
    blocks = []
    for _ in range(max_iterations):
        # Break condition: Exit if the empty string is empty
        if specification in ['', ' ']:
            return labels, blocks

        # Step 1: Encode the current specification to an embedding
        spec_embedding = encode_labels(embedding_model, [specification])

        # Step 2: Query the vector database for the nearest neighbor
        nearest_neighbors = vectordb_read(vector_db, spec_embedding, limit=1)
        if not nearest_neighbors:
            break  # Break condition: Exit if no neighbors are found

        # Step 3: Extract and store the label and block pair from the nearest neighbor
        nearest_doc = nearest_neighbors[0]
        labels.append(nearest_doc.label)
        blocks.append(nearest_doc.block)

        # Step 4: Remove common words from the label and specification, update the specification
        new_specification = remove_common_words(specification, ''.join(labels))
        if new_specification == specification: break # Break condition: Specification remains the same
        specification = new_specification # Replace the specification and start over

    return labels, blocks

def embedding_projection(current_embedding, neighbor_embedding):
    """
    Subtracts the projection of the nearest neighbor's embedding from the current embedding
    :param current_embedding: The specification's current embedding.
    :param neighbor_embedding: The nearest neighbor's embedding.
    """
    inner_product = np.dot(current_embedding, neighbor_embedding) / np.dot(neighbor_embedding, neighbor_embedding)
    return inner_product * neighbor_embedding

def linear_embedding_merge(embedding_model, vector_db, specification, max_iterations=10, norm_threshold=1e-3):
    """
    Iteratively finds the nearest neighbor to a given specification embedding,
    subtracts the projection of the nearest neighbor's embedding from the specification's
    embedding, and repeats until the embedding's norm is less than a specified threshold
    or the maximum number of iterations is reached.

    :param embedding_model: Pre-trained sentence embedding model.
    :param vector_db: Initialized vector database for storing and querying embeddings.
    :param specification: String specification to process iteratively.
    :param max_iterations: Maximum number of iterations to perform (default is 10).
    :param norm_threshold: Threshold for the norm of the embedding below which iterations stop (default is 1e-3).
    :return: List of tuples containing (label, block) for each iteration.
    """


    # Store the label and block results from each iteration
    labels = []
    blocks = []

    # Step 1: Encode the initial specification to an embedding
    current_embedding = encode_labels(embedding_model, [specification])[0]

    for _ in range(max_iterations):
        # Break condition: If the norm of the embedding is below the threshold
        if np.linalg.norm(current_embedding) < norm_threshold:
            break

        # Step 2: Query the vector database for the nearest neighbor
        nearest_neighbors = vectordb_read(vector_db, current_embedding, limit=1)
        if not nearest_neighbors:
            break  # Break condition: Exit if no neighbors are found

        # Step 3: Extract and store the label and block pair from the nearest neighbor
        nearest_doc = nearest_neighbors[0]
        labels.append(nearest_doc.label)
        blocks.append(nearest_doc.block)

        # Step 4: Get the nearest neighbor's embedding
        neighbor_embedding = encode_labels(embedding_model, [nearest_doc.label])[0]

        # Step 5: Subtract the projection of the nearest neighbor's embedding from the current embedding
        projection = embedding_projection(current_embedding, neighbor_embedding)
        if np.linalg.norm(projection) < norm_threshold: break # Break condition: Perpendicular vectors
        current_embedding = current_embedding - projection

    return labels, blocks
