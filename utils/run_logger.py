import csv
import os
from datetime import datetime

class RunLogger:
    def __init__(self, log_dir="logs", filename="contexto_runs.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, filename)

        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "game_id",
                    "model",
                    "turn",
                    "word",
                    "score",
                    "best_score"
                ])

    def log(self, game_id, model, turn, word, score, best_score):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                game_id,
                model,
                turn,
                word,
                score,
                best_score
            ])
