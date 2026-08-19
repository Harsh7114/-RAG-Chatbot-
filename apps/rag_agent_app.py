from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st





#load the fix document
loader = PyPDFLoader("../data/medical_report.pdf")
docs = loader.load()

#split into chunks 
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 200)
docs = splitter.split_documents(documents=docs)

#embeddings 
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = InMemoryVectorStore.from_documents(
    documents=docs,
    embedding=embeddings
)

# create a agent - tool , llm,prompt 
llm = ChatGroq(model="openai/gpt-oss-20b")
#tool
@tool
def retrieve_context(query:str):
    """
    Retrieve documents relevent to a query  from the knowledge base. 
    """
    print("TOOL CALLED:", query)
    context = ""
    docs = vector_db.similarity_search(query=query,k=3)
    for doc in docs:
        context += doc.page_content + "\n\n"
    return context

system_prompt= """You are a helpful assistant that answers questions using retrieved context. 
        My knowledge base consists of the details from the uploaded document. 
        ALWAYS use the `retrieve_context` tool for questions requiring external knowledge."""


memory = InMemorySaver()

agent = create_agent(
    model = llm,
    tools=[retrieve_context],
    system_prompt=system_prompt,
    checkpointer=memory
                     )


while True:
    query = input("USER: ")
    if query.lower()=="quit":
        break
    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        {"configurable":{"thread_id":1}}
    )
    result = response["messages"][-1].content
    print("AI :",result)