import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

try:
    from langchain.chains import RetrievalQA
except ImportError:
    from langchain.chains.retrieval_qa.base import RetrievalQA

def get_response(db, question):
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=db.as_retriever(),
        return_source_documents=True
    )
    
    result = qa_chain.invoke({"query": question})
    return result["result"], result["source_documents"]