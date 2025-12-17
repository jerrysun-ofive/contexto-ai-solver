# solvers/base_solver.py

from abc import ABC, abstractmethod

class BaseSolver(ABC):
    """
    Abstract class for all solver defining required interface
    """

    def __init__(self):
        self.history = []

    @abstractmethod
    def get_next_guess(self, history):
        """Return the next guess string."""
        pass

    def update_state(self, guess: str, score: int):
        """Store guess + score into history."""
        self.history.append((guess, score))
