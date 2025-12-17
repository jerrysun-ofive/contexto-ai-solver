import numpy as np
import random
import re

from solvers.base_solver import BaseSolver
from utils.similarity import cosine_similarity
from utils.vector_utils import load_embeddings

# def global starting word
STARTING_WORD = "human"

class Word2VecSolver(BaseSolver):
    model_name = "word2vec-v1"
    def __init__(
            self,
            top_k: int = 8,
            stall_window: int = 8,
            jump_pool: int = 300,
        ):
        super().__init__()
        vocabFile = "models/v1_word2vec/gensim_vocab.npy"
        embeddingFile = "models/v1_word2vec/gensim_embeddings.npy"

        self.vocab, self.embeddings, self.word_to_idx = load_embeddings(vocabFile, embeddingFile)
        
        # clean up junk token
        self.clean_vocab = [
            w for w in self.vocab
            if re.fullmatch(r"[a-z]+", w) and 3 <= len(w) <= 20
        ]

        # pre comp global mean (centre of embedding space)
        self.global_mean = np.mean(
            [self._vector(w) for w in self.clean_vocab[:5000]],
            axis=0
        )

        # track already guessed words to avoid inifite loop since w2v is very symmetric
        # 2 cycle local optimum
        # solver state
        self.history = []
        self.guessed = set()
        self.best_score_seen = float("inf")
        self.last_improve_turn = -1

        # hyperparameters
        self.top_k = top_k
        self.stall_window = stall_window
        self.jump_pool = jump_pool

        print(
            f"[Word2VecSolver] vocab={len(self.vocab)} "
            f"clean_vocab={len(self.clean_vocab)}"
        )

# ---------------------------------------------------------------------------- #
#                                     utils                                    #
# ---------------------------------------------------------------------------- #

    def _vector(self, word):
        idx = self.word_to_idx.get(word)
        if idx is None:
            return None

        return self.embeddings[idx]
    
# ---------------------------------------------------------------------------- #
#                                  solver api                                  #
# ---------------------------------------------------------------------------- #
    
    def get_next_guess(self, history):
        """
        Approach:
        - If not history, guess random word
        - else: pick the word whose embedding is closest to the last guess
        """
        # first guess
        if not history:
            if STARTING_WORD in self.word_to_idx:
                self.guessed.add(STARTING_WORD)
                return STARTING_WORD
            else:
                guess = random.choice(self.clean_vocab)
                self.guessed.add(guess)
                return guess
        
        turn = len(history)
        
        last_word, last_score = history[-1]
        if last_score < self.best_score_seen:
            self.best_score_seen = last_score
            self.last_improve_turn = turn - 1

        # find best and worst guess so far
        best = sorted(history, key=lambda x: x[1])[:self.top_k]
        worst = sorted(history, key=lambda x: x[1], reverse=True)[:self.top_k]

        best_vec = np.mean([self._vector(w) for w, _ in best], axis=0)
        worst_vec = np.mean([self._vector(w) for w, _ in worst], axis=0)

        # stall detection
        stalled = (turn - self.last_improve_turn) > self.stall_window

        if stalled:
            # do a guided explore jump
            pool = []
            while len(pool) < self.jump_pool:
                w = random.choice(self.clean_vocab)
                if w not in self.guessed:
                    pool.append(w)

            def jump_objective(w):
                vw = self._vector(w)
                return (
                    cosine_similarity(best_vec, vw)
                    - 0.7 * cosine_similarity(worst_vec, vw)
                    - 0.3 * cosine_similarity(self.global_mean, vw)
                )

            guess = max(pool, key=jump_objective)
            self.guessed.add(guess)
            return guess
    
        # normal step
        direction = best_vec - worst_vec - 0.3 * self.global_mean

        pool = []
        while len(pool) < 2000:
            w = random.choice(self.clean_vocab)
            if w not in self.guessed:
                pool.append(w)

        guess = max(
            pool,
            key=lambda w: cosine_similarity(direction, self._vector(w))
        )

        self.guessed.add(guess)
        return guess

    def update_state(self, guess: str, score: int):
        """
        Called after each guess with the returned Contexto score.
        """
        self.history.append((guess, score))

    def reset(self):
        self.history = []
        self.guessed = set()
        self.best_score_seen = float("inf")
        self.last_improve_turn = -1
