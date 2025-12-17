from scraping.fetch_contexto_page import get_contexto_page
from scraping.contexto_web_interface import submit_guess, get_latest_score

# interface for solvers
from solvers.base_solver import BaseSolver
from solvers.solver_gensim_w2v import Word2VecSolver

# util
from utils.run_logger import RunLogger
import uuid

def debug_ranks(page):
    page.wait_for_timeout(2000)
    rows = page.query_selector_all(".rank")
    print("Rank elements found:", len(rows))
    for i, r in enumerate(rows):
        print(i, repr(r.inner_html()))

def debug_after_guess(page):
    page.wait_for_timeout(2000)

    # print last 5 rows of the guesses list
    rows = page.query_selector_all("div")
    print("Total divs:", len(rows))

    for r in rows[-20:]:
        html = r.inner_text().strip()
        if html:
            print("TEXT:", repr(html))

def play_contexto(solver: BaseSolver, max_attempt: int=100):
    INVALID_WORD_PENALTY = 100_000
    browser, page = get_contexto_page(headless=False)
    print(page.frames)

    history = []
    best_score = float("inf")

    print(
        f"Starting solver"
    )

    history = []
    print("Starting solver")

    for turn in range(max_attempt):
        # ask for next guess
        guess = solver.get_next_guess(history)
        print(f"\nTurn {turn+1}: Guess -> {guess}")

        # submit guess
        submit_guess(page, guess)
        # debug_ranks(page)
        # debug_after_guess(page)
        score = get_latest_score(page, guess)

        # get score
        if score is None:
            print(f"Invalid guess (not recognised): {guess}")
            score = INVALID_WORD_PENALTY
            continue
        
        # logger
        best_score = min(best_score, score)
        print(f"Score: {score} | Best so far: {best_score}")

        # update solver states
        history.append((guess, score))
        solver.update_state(guess, score)

        # if solved
        if score == 1:
            print("Solved. The word is: ", guess)
            break

    browser.close()

if __name__ == "__main__":
    print("Playing Contexto")
    solver = Word2VecSolver()
    solver.reset()
    play_contexto(solver)