!pip install -q pyppeteer nest_asyncio
import asyncio
import nest_asyncio
from pyppeteer import launch
import os

# Fix event loop issue in Colab
nest_asyncio.apply()

html_dir = "/content/drive/MyDrive/plots"
png_dir = "/content/drive/MyDrive/plots_png"
os.makedirs(png_dir, exist_ok=True)

async def convert_tvalue_htmls():
    browser = await launch(headless=True, args=["--no-sandbox"])
    page = await browser.newPage()

    for file in os.listdir(html_dir):
        if file.endswith("_t_value.html"):   # only t_value plots
            html_path = os.path.join(html_dir, file)
            png_path = os.path.join(png_dir, file.replace(".html", ".png"))

            print(f"Converting {file} → {png_path}")
            await page.goto(f"file://{html_path}")
            await page.screenshot({"path": png_path, "fullPage": True})

    await browser.close()

await convert_tvalue_htmls()
