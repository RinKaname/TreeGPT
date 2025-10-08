import asyncio
from playwright.async_api import async_playwright
import os

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        # Construct the file path to the index.html
        # The script is run from the root, so the path is relative to the root
        file_path = os.path.abspath('docs/_build/html/index.html')

        # Go to the local HTML file
        await page.goto(f'file://{file_path}')

        # Take a screenshot
        await page.screenshot(path='jules-scratch/verification/docs_verification.png')

        await browser.close()

if __name__ == '__main__':
    asyncio.run(main())