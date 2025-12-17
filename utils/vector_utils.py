
import numpy as np

def load_embeddings(vocabFilePath, embeddingsFilePath):
    vocab = np.load(vocabFilePath, allow_pickle=True)
    embeddings = np.load(embeddingsFilePath)

    word_to_idx = {word: i for i, word in enumerate(vocab)}
    
    return vocab, embeddings, word_to_idx