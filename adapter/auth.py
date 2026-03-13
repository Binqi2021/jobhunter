from playwright.sync_api import sync_playwright

def save_auth():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        page.goto("https://www.nowcoder.com/jobs/center")

        page.wait_for_timeout(30000)
        context.storage_state(path="auth.json")

        browser.close()

if __name__ == "__main__":
    save_auth()