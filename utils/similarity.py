import numpy as np

def cosine_similarity(a, b):
    """
    Computes cosine similarity for two vectors.
    """
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))