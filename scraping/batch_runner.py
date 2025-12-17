import time
import uuid

from tqdm import tqdm

from scraping.fetch_contexto_page import get_contexto_page
from scraping.contexto_web_interface import submit_guess, get_latest_score

from solvers.solver_gensim_w2v import Word2VecSolver
from utils.run_logger import RunLogger


INVALID_WORD_PENALTY = 100_000


def run_single_game(page, solver, logger, max_attempts=50):
    game_id = str(uuid.uuid4())[:8]

    solver.reset()

    history = []
    best_score = float("inf")

    for turn in range(max_attempts):
        guess = solver.get_next_guess(history)

        submit_guess(page, guess)
        score = get_latest_score(page, guess)

        if score is None:
            score = INVALID_WORD_PENALTY

        best_score = min(best_score, score)

        logger.log(
            game_id=game_id,
            model=solver.model_name,
            turn=turn + 1,
            word=guess,
            score=score,
            best_score=best_score,
        )

        history.append((guess, score))
        solver.update_state(guess, score)

        if score == 1:
            break

    return best_score


def run_batch(
    num_games=30,
    max_attempts=50,
    cooldown_sec=0.8,
):
    logger = RunLogger()
    solver = Word2VecSolver()

    browser, page = get_contexto_page(headless=True)

    results = []

    try:
        with tqdm(
            total=num_games,
            desc=f"Running Contexto ({solver.model_name})",
            unit="game",
            dynamic_ncols=True,
        ) as pbar:

            for _ in range(num_games):
                page.goto("https://contexto.me/en/")
                page.wait_for_timeout(800)

                best = run_single_game(
                    page,
                    solver,
                    logger,
                    max_attempts=max_attempts,
                )

                results.append(best)

                pbar.set_postfix({
                    "last_best": best,
                    "avg_best": f"{sum(results) / len(results):.1f}",
                })

                pbar.update(1)
                time.sleep(cooldown_sec)

    finally:
        browser.close()

    print("\nBatch finished.")
    print(f"Games run: {len(results)}")
    print(f"Average best score: {sum(results) / len(results):.1f}")
    print(f"Best overall score: {min(results)}")


if __name__ == "__main__":
    run_batch(
        num_games=30,
        max_attempts=50,
        cooldown_sec=0.8,
    )
