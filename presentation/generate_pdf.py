import os
import asyncio
from playwright.async_api import async_playwright


async def generate_pdf():
    # Get absolute path to the HTML file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(current_dir, "final_presentation.html")
    pdf_path = os.path.join(current_dir, "idarati_presentation.pdf")
    file_url = f"file://{html_path}"

    print(f"Generating PDF from: {file_url}")

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        # Open the presentation
        await page.goto(file_url, wait_until="networkidle")

        # Force landscape size to ensure responsive grids look right
        await page.set_viewport_size({"width": 1920, "height": 1080})

        # Extra wait to ensure iframes are fully rendered
        await asyncio.sleep(2)

        # Generate PDF
        # We use custom width/height to match 16:9 aspect ratio (Landscape)
        # 297mm x 210mm is A4 Landscape.
        # For screen-like PDF, we define a custom page size.
        await page.pdf(
            path=pdf_path,
            width="1280px",
            height="720px",
            print_background=True,
            margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
            page_ranges="",  # All pages
        )

        print(f"✅ Success! PDF saved to: {pdf_path}")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(generate_pdf())
