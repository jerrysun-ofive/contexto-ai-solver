from typing import Optional
from playwright.sync_api import Page
import re

# submit a guess into the Contexto input field
def submit_guess(page: Page, word: str):
    input_box = page.query_selector("input[type='text']")
    input_box.fill(word)
    input_box.press("Enter")

# get the similarity score of most recent guess
def get_latest_score(page: Page, guess: str) -> Optional[int]:
    page.wait_for_timeout(800)

    body = page.inner_text("body")

    # Contexto renders:
    # word\nNUMBER
    pattern = rf"\b{re.escape(guess[:-1])}\w*\b\s*\n\s*(\d+)"
    matches = re.findall(pattern, body, re.IGNORECASE)

    if not matches:
        return None

    # take the LAST occurrence for THIS WORD
    score = int(matches[-1])
    # print(f"raw score: {score}")
    return score

