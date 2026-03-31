import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

def generate_assessment(db):
    docs = db.get(limit=3)
    context = "\n".join(docs['documents'])
    
    llm = ChatOpenAI(temperature=0.4, model_name="gpt-3.5-turbo")
    
    template = """Imagine that You are a Professor.Create a 3-question MCQ quiz based on this context. 
    Output ONLY a valid JSON list of objects.
    Context: {context}"""
    
    prompt = PromptTemplate(template=template, input_variables=["context"])
    response = llm.invoke(prompt.format(context=context))
    
    return json.loads(response.content) # Returns a Python List