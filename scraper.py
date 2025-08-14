import asyncio
import aiohttp
from bs4 import BeautifulSoup
from langchain.schema import Document
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def fetch_and_create_document(session, url):
    """Fetches a single URL, parses it, and creates a Document."""
    try:
        async with session.get(url, timeout=10) as response:
            response.raise_for_status()
            html = await response.text()
            
            soup = BeautifulSoup(html, 'html.parser')
            
            article_content = soup.find('article')
            if article_content:
                text = article_content.get_text(separator=' ', strip=True)
            else:
                text = soup.get_text(separator=' ', strip=True)

            header_marker = "Главная / Портфолио /"
            if header_marker in text:
                text = text.split(header_marker, 1)[1]

            footer_marker = '[{"lid"'
            if footer_marker in text:
                text = text.split(footer_marker, 1)[0]

            text = text.strip()

            if text:
                logging.info(f"Successfully scraped {url}")
                return Document(page_content=text, metadata={"source": url})
            else:
                logging.warning(f"No content scraped from {url}")
                return None

    except aiohttp.ClientError as e:
        logging.error(f"Error fetching {url}: {e}")
        return None
    except Exception as e:
        logging.error(f"An error occurred while processing {url}: {e}")
        return None

async def scrape_links():
    """
    Main function to read URLs and run scraping tasks concurrently.
    """
    links_file = 'links.txt'
    try:
        with open(links_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logging.error(f"Error: {links_file} not found.")
        return []

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_and_create_document(session, url) for url in urls]
        documents = await asyncio.gather(*tasks)
        return [doc for doc in documents if doc]

if __name__ == '__main__':
    documents = asyncio.run(scrape_links())
    logging.info(f"Scraped {len(documents)} documents.")