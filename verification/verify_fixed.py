import time
from playwright.sync_api import sync_playwright, expect

def verify_fixed():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print("Navigating to app...")
        page.goto("http://localhost:7860")
        page.wait_for_load_state("networkidle")

        # Take initial screenshot
        print("Taking initial screenshot...")
        page.screenshot(path="verification/fixed_ui.png", full_page=True)

        # Verify Status is ABOVE Preview
        preview_label = page.get_by_text("Preview Output") # Accordion label
        status_label = page.get_by_text("Status", exact=True).first

        if preview_label.count() > 0 and status_label.count() > 0:
            preview_y = preview_label.bounding_box()['y']
            status_y = status_label.bounding_box()['y']
            print(f"Status Y: {status_y}, Preview Y: {preview_y}")
            if status_y < preview_y:
                print("Confirmed: Status is ABOVE Preview.")
            else:
                print("FAILED: Status is NOT above Preview.")

        # Verify Examples exist
        if page.get_by_text("Examples").count() > 0:
            print("Confirmed: Examples section present.")
        else:
            print("FAILED: Examples section missing.")

        # Switch to Custom List mode
        print("Switching to ImageNet (Custom List)...")
        page.get_by_text("ImageNet (Custom List)").click()

        # Click Preview with empty input
        print("Clicking Preview with empty input...")
        page.get_by_role("button", name="Preview YAML").click()

        time.sleep(2)

        status_box = page.get_by_label("Status")
        status_value = status_box.input_value()
        print(f"Status Value: {status_value}")

        if "❌ Error: No WNIDs provided" in status_value:
             print("Confirmed: Correct error message in status.")
        else:
             print("FAILED: Incorrect error message.")

        # Capture final state
        page.screenshot(path="verification/fixed_error.png", full_page=True)

        browser.close()

if __name__ == "__main__":
    verify_fixed()
