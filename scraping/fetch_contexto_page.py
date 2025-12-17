from playwright.sync_api import sync_playwright

URL = "https://contexto.me/en/"

BLOCKED_RESOURCE_TYPES = {
    "image",
    "font",
    "media",
}

BLOCKED_DOMAINS = [
    "doubleclick.net",
    "googlesyndication.com",
    "google-analytics.com",
    "adsystem.com",
    "facebook.net",
    "analytics",
]

def get_contexto_page(headless: bool = True):
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=headless)
    page = browser.new_page()

    # block ads based on predefined popular domains
    def block_ads(route, request):
        url = request.url
        if (
            request.resource_type in BLOCKED_RESOURCE_TYPES
            or any(domain in url for domain in BLOCKED_DOMAINS)
        ):
            route.abort()
        else:
            route.continue_()

    page.route("**/*", block_ads)

    page.set_default_navigation_timeout(120_000)
    page.goto(URL, wait_until="domcontentloaded")
    page.wait_for_selector("input[type='text']", timeout=120_000)

    return browser, page
