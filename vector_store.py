import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_PATH = "data"
CHROMA_PATH = "chromadb"
MODEL_NAME = "intfloat/multilingual-e5-large"
EMBEDDINGS = HuggingFaceEmbeddings(model_name=MODEL_NAME)


def create_vector_store():
    """
    Creates a Chroma vector store from documents in the data directory.
    """
    print("Loading documents...")
    loader = DirectoryLoader(DATA_PATH, glob="*.txt", show_progress=True, use_multithreading=True)
    documents = loader.load()

    if not documents:
        print("No documents found in the 'data' directory. Please run the scraper first.")
        return

    print(f"Loaded {len(documents)} document(s).")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")

    print(f"Initializing Hugging Face Embeddings with model: {MODEL_NAME}...")
    print("This may take a moment as the model needs to be downloaded on the first run.")

    print("Creating and persisting vector store...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=EMBEDDINGS,
        persist_directory=CHROMA_PATH
    )

    print(f"Vector store created successfully and saved to '{CHROMA_PATH}'.")
    print(f"Total vectors in store: {vector_store._collection.count()}")


if __name__ == '__main__':
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print(f"The '{DATA_PATH}' directory is missing or empty.")
        print("Please run the scraper.py script first to gather data.")
    else:
        create_vector_store()
