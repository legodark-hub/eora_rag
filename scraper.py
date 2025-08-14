import os
import asyncio
import aiohttp
import aiofiles
from bs4 import BeautifulSoup
import re

def get_filename_from_url(url):
    """Generates a clean filename from a URL."""
    if not url:
        return None
    sanitized_part = re.sub(r'[^a-zA-Z0-9_\-]', '', url.split('/')[-1])
    return f"{sanitized_part}.txt" if sanitized_part else None

async def fetch_and_save(session, url, output_dir):
    """Fetches a single URL, parses it, and saves the content."""
    try:
        async with session.get(url, timeout=10) as response:
            response.raise_for_status()
            html = await response.text()
            
            soup = BeautifulSoup(html, 'html.parser')
            
            article_content = soup.find('article')
            if article_content:
                text = article_content.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)

            filename = get_filename_from_url(url)
            if filename:
                filepath = os.path.join(output_dir, filename)
                async with aiofiles.open(filepath, 'w', encoding='utf-8') as out_file:
                    await out_file.write(f"Source: {url}\n\n")
                    await out_file.write(text)
                print(f"Successfully scraped and saved {url} to {filepath}")
            else:
                print(f"Could not generate a valid filename for URL: {url}")

    except aiohttp.ClientError as e:
        print(f"Error fetching {url}: {e}")
    except Exception as e:
        print(f"An error occurred while processing {url}: {e}")

async def main():
    """
    Main function to read URLs and run scraping tasks concurrently.
    """
    links_file = 'links.txt'
    output_dir = 'data'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(links_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: {links_file} not found.")
        return

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_and_save(session, url, output_dir) for url in urls]
        await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(main())