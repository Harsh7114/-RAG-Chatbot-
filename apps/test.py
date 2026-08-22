import os
import uuid
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# --------------------------------------------------
# PROCESS DOCUMENT
# --------------------------------------------------

def process_document(uploaded_files):

    # Create temporary directory
    path = "./docs_files"

    os.makedirs(path, exist_ok=True)

    # Remove old PDFs
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)

        if os.path.isfile(file_path):
            os.remove(file_path)

    # Save only currently uploaded files
    for file in uploaded_files:

        file_path = os.path.join(path, file.name)

        with open(file_path, "wb") as f:
            f.write(file.getvalue())

    # --------------------------------------------------
    # LOAD PDFs
    # --------------------------------------------------

    loader = PyPDFDirectoryLoader(path)

    docs = loader.load()

    if not docs:
        st.error("No readable PDF content found.")
        return

    # --------------------------------------------------
    # SPLIT DOCUMENTS
    # --------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    docs = splitter.split_documents(docs)

    # --------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # --------------------------------------------------
    # VECTOR STORE
    # --------------------------------------------------

    vector_store = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )

    st.session_state.vector_store = vector_store

    # --------------------------------------------------
    # LLM
    # --------------------------------------------------

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=st.secrets["GROQ_API_KEY"],
        temperature=0
    )

    # --------------------------------------------------
    # RETRIEVAL TOOL
    # --------------------------------------------------

    @tool
    def retrieve_context(query: str) -> str:
        """
        Retrieve relevant information from the uploaded PDF documents.
        Only use this tool for questions about the uploaded documents.
        """

        print("TOOL CALLED:", query)

        results = vector_store.similarity_search_with_score(
            query,
            k=5
        )

        # Keep only sufficiently relevant results
        relevant_docs = []

        for doc, score in results:

            print("SCORE:", score)

            # Lower score generally means more similar
            if score < 1.0:
                relevant_docs.append(doc)

        # No sufficiently relevant information
        if not relevant_docs:
            return "NO_RELEVANT_INFORMATION_FOUND"

        context = ""

        for doc in relevant_docs:

            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")

            context += (
                f"\n--- SOURCE: {source}, PAGE: {page} ---\n"
                f"{doc.page_content}\n"
            )

        return context

    # --------------------------------------------------
    # SYSTEM PROMPT
    # --------------------------------------------------

    system_prompt = """
You are a PDF question-answering assistant.

Your ONLY source of factual information is the uploaded PDF document.

IMPORTANT RULES:

1. ALWAYS use the retrieve_context tool for questions about the uploaded PDF.

2. Answer ONLY using information returned by the retrieve_context tool.

3. NEVER use your own pretrained knowledge to fill missing information.

4. If the retrieved context does not contain the answer, respond exactly:

"I couldn't find this information in the uploaded PDF."

5. Do not guess.

6. Do not invent names, projects, skills, dates, links, qualifications,
companies, technologies, or any other information.

7. If the user asks a question unrelated to the uploaded PDF, say:

"I can only answer questions based on the uploaded PDF."

8. Keep answers concise and directly related to the question.

9. If multiple pieces of information are available in the PDF,
combine them accurately.

10. When possible, mention the page number from the retrieved context.
"""

    # --------------------------------------------------
    # MEMORY
    # --------------------------------------------------

    memory = InMemorySaver()

    # --------------------------------------------------
    # CREATE AGENT
    # --------------------------------------------------

    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True

    # Clear previous chat
    st.session_state.messages = []

    # New conversation ID
    st.session_state.thread_id = str(uuid.uuid4())


# --------------------------------------------------
# UPLOAD UI
# --------------------------------------------------

if not st.session_state.document_uploaded:

    st.title("📄 PDF RAG Chatbot")

    uploaded = st.file_uploader(
        "Upload PDF file(s)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded:

        with st.spinner("Processing PDF..."):

            process_document(uploaded)

        st.success("PDF processed successfully!")

        st.rerun()


# --------------------------------------------------
# CHAT UI
# --------------------------------------------------

if st.session_state.document_uploaded:

    st.title("💬 Ask Questions About Your PDF")

    # Reset button
    if st.button("🔄 Upload New PDF"):

        st.session_state.document_uploaded = False
        st.session_state.agent = None
        st.session_state.vector_store = None
        st.session_state.messages = []

        st.rerun()

    # --------------------------------------------------
    # DISPLAY OLD MESSAGES
    # --------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------

    query = st.chat_input(
        "Ask anything related to uploaded documents..."
    )

    if query:

        # Add user message immediately
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        # Display user message
        with st.chat_message("user"):
            st.markdown(query)

        # Generate response
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    response = st.session_state.agent.invoke(
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": query
                                }
                            ]
                        },
                        {
                            "configurable": {
                                "thread_id": st.session_state.thread_id
                            }
                        }
                    )

                    answer = response["messages"][-1].content

                except Exception as e:

                    answer = (
                        "Sorry, I encountered an error while "
                        "processing your question."
                    )

                    st.error(str(e))

                st.markdown(answer)

        # Save assistant response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )