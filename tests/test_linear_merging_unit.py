import pytest, sys
sys.path.append("..")
from llm_blockmerger.block_merge import *

@pytest.fixture
def projection_data():
    return [
    (np.array([0, 0]), np.array([1, 0]), None),  # Case: neighbor_embedding is zero
    (np.array([1, 2]), np.array([2, 4]), np.array([2.0, 4.0])),  # Case: collinear vectors
    (np.array([3, 4]), np.array([4, 3]), np.array([2.88, 3.84])),  # Case: non-collinear vectors
    (np.array([1, 1]), np.array([-1, -1]), np.array([-1.0, -1.0])),  # Case: opposite direction
    (np.array([1,0]), np.array([0,1]), np.array([0.0, 0.0])) # Case: perpendicular vectors
]

def test_embedding_projection(projection_data):
    for current_embedding, neighbor_embedding, expected_embedding in projection_data:
        projections = embedding_projection(current_embedding, neighbor_embedding)
        assert np.all(projections == expected_embedding)


@pytest.fixture
def remove_words_data():
    return [
        # Case: Common words replaced
        ("This is a test", "is a", "UNKNOWN", "This UNKNOWN UNKNOWN test"),

        # Case: Case insensitivity
        ("This Is a Test", "is A", "UNKNOWN", "This UNKNOWN UNKNOWN Test"),

        # Case: No words to remove
        ("This is a test", "", "UNKNOWN", "This is a test"),

        # Case: All words removed
        ("This is a test", "this is a test", "UNKNOWN", "UNKNOWN UNKNOWN UNKNOWN UNKNOWN"),

        # Case: Special characters in original
        ("Hello, world! This is a test.", "is a", "UNKNOWN", "Hello, world! This UNKNOWN UNKNOWN test."),

        # Case: Replacement string other than "UNKNOWN"
        ("This is a test", "is a", "REMOVED", "This REMOVED REMOVED test"),

        # Case: Original string is empty
        ("", "is a", "UNKNOWN", ""),

        # Case: Words to remove not present
        ("This is a test", "missing words", "UNKNOWN", "This is a test"),

        # Case: Original string has extra spaces
        ("  This   is  a    test   ", "is a", "UNKNOWN", "This UNKNOWN UNKNOWN test"),
    ]


def test_remove_common_words(remove_words_data):
    for original, to_remove, replacement, expected in remove_words_data:
        result = remove_common_words(original, to_remove, replacement)
        assert result == expected

