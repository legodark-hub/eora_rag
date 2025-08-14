import asyncio
import os
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from scraper import scrape_links

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CHROMA_PATH = "chromadb"
MODEL_NAME = "intfloat/multilingual-e5-large"
EMBEDDINGS = HuggingFaceEmbeddings(model_name=MODEL_NAME)

def create_vector_store_if_not_exists():
    """
    Creates the vector store if it doesn't exist.
    """
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        logging.info("Vector store already exists. Skipping creation.")
        return

    logging.info("Vector store not found. Creating...")
    documents = asyncio.run(scrape_links())
    
    if not documents:
        logging.warning("No documents scraped. Vector store not created.")
        return

    logging.info(f"Loaded {len(documents)} document(s).")

    logging.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split documents into {len(chunks)} chunks.")

    logging.info(f"Initializing Hugging Face Embeddings with model: {MODEL_NAME}...")
    logging.info("This may take a moment as the model needs to be downloaded on the first run.")

    logging.info("Creating and persisting vector store...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=EMBEDDINGS,
        persist_directory=CHROMA_PATH
    )

    logging.info(f"Vector store created successfully and saved to '{CHROMA_PATH}'.")
    logging.info(f"Total vectors in store: {vector_store._collection.count()}")


if __name__ == '__main__':
    create_vector_store_if_not_exists()