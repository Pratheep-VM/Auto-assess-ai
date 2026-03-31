# --- BLOCK 1: IMPORTS 
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma 

# --- BLOCK 2: SETUP ---
load_dotenv()

# --- Define a persistent directory for our vector database ---
# This will create a folder named 'db' in your project directory 
# where the vector database files will be stored.
PERSIST_DIRECTORY = "./db"

def process_and_store_document(file_path: str):
    # --- Part 1: Loading and Splitting ---
    print(f"Loading document: {file_path}...")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"Successfully loaded {len(pages)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    print("Splitting text into chunks...")
    chunks = text_splitter.split_documents(pages)
    print(f"Successfully split the document into {len(chunks)} chunks.")
    
    # --- Part 2: Creating Embeddings and Storing in Vector DB (The NEW part) ---
    print("Creating embeddings and storing them in the vector database...")
    
    # 2a. Initialize the Embedding Model
    # This tells LangChain to use HuggingFace's model to turn text into vectors.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2b. Create the Vector Database from our chunks
    # LangChain takes our chunks, uses the embedding model
    # to get a vector for each one, and then stores the chunk and its vector
    # together in the ChromaDB database in our 'db' folder.
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    
    # We explicitly tell the database to save itself to the disk.
    vectordb.persist()
    
    print(f"Successfully created and saved the vector database at: {PERSIST_DIRECTORY}")

# --- BLOCK 4: RUNNING THE SCRIPT ---
if __name__ == "__main__":
    # We call our new function.
    process_and_store_document("Mathematics_for_ML.pdf")