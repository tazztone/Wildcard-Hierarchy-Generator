import time
from playwright.sync_api import sync_playwright, expect

def verify_current():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print("Navigating to app...")
        page.goto("http://localhost:7860")
        page.wait_for_load_state("networkidle")

        # Take initial screenshot
        print("Taking initial screenshot...")
        page.screenshot(path="verification/current_ui.png", full_page=True)

        # Verify Status is below Preview
        # Get elements
        # Preview label is "Preview" (Code component)
        # Status label is "Status" (Textbox component)

        # Gradio structure: span with text "Preview", then the code block.
        # We can find the labels.
        preview_label = page.get_by_text("Preview", exact=True).first # might match button "Preview YAML"
        status_label = page.get_by_text("Status", exact=True).first

        # Check vertical position
        if preview_label.count() > 0 and status_label.count() > 0:
            preview_y = preview_label.bounding_box()['y']
            status_y = status_label.bounding_box()['y']
            print(f"Preview Y: {preview_y}, Status Y: {status_y}")
            if status_y > preview_y:
                print("Confirmed: Status is below Preview.")
            else:
                print("Unexpected: Status is NOT below Preview.")

        # Switch to Custom List mode
        print("Switching to ImageNet (Custom List)...")
        # Click the text "ImageNet (Custom List)" which should be part of the label
        page.get_by_text("ImageNet (Custom List)").click()

        # Click Preview without entering text
        print("Clicking Preview with empty input...")
        page.get_by_role("button", name="Preview YAML").click()

        # Wait for result - expecting failure or error
        time.sleep(2)

        # Check if error appeared in status or if page crashed
        # If app crashed, page might be unresponsive or show error.

        status_box = page.get_by_label("Status")
        status_value = status_box.input_value()
        print(f"Status Value: {status_value}")

        # Capture error state
        page.screenshot(path="verification/current_error.png", full_page=True)

        browser.close()

if __name__ == "__main__":
    verify_current()
