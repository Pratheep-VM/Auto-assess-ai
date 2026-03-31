import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PERSIST_DIRECTORY = "./db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def ingest_document(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # Create and return the database object
    vectordb = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=PERSIST_DIRECTORY
    )
    return vectordb

# This allows you to still run this file by itself for testing
if __name__ == "__main__":
    ingest_document("Mathematics_for_ML.pdf")
    print("Ingestion complete!")