"""
Train a Word2Vec model (Gensim) and export:
- vocab.npy
- embeddings.npy
"""

from gensim.models import Word2Vec
import numpy as np
import os

STOPWORDS = {
    "the","of","and","to","a","in","is","it","that","for","on",
    "with","as","are","was","be","by","this","an","or","from"
}

def load_corpus(chunk_size=100):
    """
    Load text8 and return List[List[str]] suitable for Word2Vec with stopwords removed.
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path = os.path.join(root, "data", "raw", "text8.txt")

    print("Loading corpus from:", file_path)
    with open(file_path, "r") as f:
        tokens = f.read().split()

    # remove stopwords
    tokens = [t for t in tokens if t not in STOPWORDS]

    # chunk into pseudo-sentences
    sentences = [
        tokens[i:i + chunk_size]
        for i in range(0, len(tokens), chunk_size)
    ]

    print(f"Total sentences: {len(sentences)}")
    return sentences


def train_word2vec():
    """ train a word2vec model using the gensim library """
    MODEL_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "models",
        "v1_word2vec"
    )
    os.makedirs(MODEL_DIR, exist_ok=True)

    
    print("Training Gensim Word2Vec model")
    corpus = load_corpus()

    model = Word2Vec(
            sentences=corpus,
            vector_size=100,        # embedding dimensions
            window=5,               # context window size
            workers=4,              # parallel threads
            epochs=10,              # number of training iteration over the corpus
            min_count=5,            # ignore rare words
            sg=1,                   # use skip-gram which is better for semantics
            negative=10,            # stronger negative sampling
        )
    
    model_path = os.path.join(MODEL_DIR, "gensim_word2vec.model")
    model.save(model_path)
    print("Model saved:", model_path)

    vocab = list(model.wv.index_to_key)
    embeddings = np.array([model.wv[word] for word in vocab])

    np.save(os.path.join(MODEL_DIR, "gensim_vocab.npy"),
            np.array(vocab, dtype=object))
    np.save(os.path.join(MODEL_DIR, "gensim_embeddings.npy"),
            embeddings)

    print("Saved vocab & embeddings to:", MODEL_DIR)


if __name__ == "__main__":
    train_word2vec()
