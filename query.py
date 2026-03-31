import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# --- SETUP ---
# Load the API key from the .env file.
load_dotenv()

# Define the paths for our components.
PERSIST_DIRECTORY = "./db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def main():
    # --- 1. INITIALIZE THE COMPONENTS ---

    # Initialize the same embedding model we used for ingestion.
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Load the existing vector database from disk.
    print("Loading vector database...")
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

    # Initialize the Large Language Model (LLM).
    # OpenAI's gpt-3.5-turbo is used, which is powerful and cost-effective.
    print("Loading Large Language Model...")
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

    # --- 2. CREATE THE RETRIEVALQA CHAIN ---
    print("Setting up retriever...")
    retriever = db.as_retriever()
    
    # The RetrievalQA chain combines the LLM and the retriever.
    # It will:
    #   1. Take your question.
    #   2. Find relevant documents in the DB using the retriever.
    #   3. "Stuff" those documents and your question into a prompt for the LLM.
    #   4. Return the LLM's answer.
    print("Creating RetrievalQA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True # This allows us to see which chunks were used.
    )

    print("--- Ready to answer your questions! ---")
    
    # --- 3. ASK QUESTIONS IN A LOOP ---
    while True:
        query = input("\nAsk a question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        # Send the query to our chain.
        result = qa_chain.invoke({"query": query})
        
        # Print the final answer.
        print("\n> Answer:")
        print(result["result"])

        # (Optional) Print the source documents to see where the answer came from.
        print("\n> Sources:")
        for source in result["source_documents"]:
            print(f"- {source.metadata['source']} (Page {source.metadata['page']})")


if __name__ == "__main__":
    main()