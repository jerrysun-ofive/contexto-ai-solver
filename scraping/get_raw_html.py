from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from bs4 import BeautifulSoup
import os

options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)

driver.get("https://contexto.me/en/")
time.sleep(3)  # wait for JS to build the page

html = driver.page_source

# prettify
soup = BeautifulSoup(html, "html.parser")
pretty_html = soup.prettify()

# write result into a file
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "raw_html_page.html")
with open(file_path, "w", encoding="utf-8") as f:
    f.write(pretty_html)

driver.quit()

print("Saved rendered page with all elements.")


