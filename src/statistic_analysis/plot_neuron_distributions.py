!pip install nest_asyncio

import nest_asyncio
import asyncio
import os
from pyppeteer import launch

nest_asyncio.apply()  # allow nested event loops (needed in Colab)

async def htmls_to_pngs(input_dir="/content/drive/MyDrive/plots",
                        output_dir="/content/drive/MyDrive/plots/pngs"):
    os.makedirs(output_dir, exist_ok=True)
    browser = await launch(headless=True, args=['--no-sandbox'])
    page = await browser.newPage()
    await page.setViewport({'width': 1920, 'height': 1080})

    for fname in os.listdir(input_dir):
        if fname.endswith(".html"):
            html_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname.replace(".html", ".png"))

            await page.goto("file://" + html_path)
            await asyncio.sleep(2)  # wait for plotly JS to render
            await page.screenshot({'path': out_path, 'fullPage': True})
            print(f"Converted: {fname} → {out_path}")

    await browser.close()

# Run in Colab properly
await htmls_to_pngs()
