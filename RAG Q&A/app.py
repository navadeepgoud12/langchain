import os
import time
import streamlit as st

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_classic.chains.retrieval import (
    create_retrieval_chain,
)

# ---------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# ---------------------------------------------------
# LLM
# ---------------------------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key,
)

# ---------------------------------------------------
# Prompt
# ---------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are an expert AI assistant.

Responsibilities:
- Answer only from the provided context.
- If the answer is not present in the context, say:
"I don't have enough information to answer that."
- Never make up information.
- Explain step by step whenever appropriate.
- Use markdown formatting.

Context:
{context}

Question:
{input}

Answer:
"""
)

# ---------------------------------------------------
# Streamlit
# ---------------------------------------------------

st.title("📄 RAG Document Q&A with Groq + Llama 3")

# ---------------------------------------------------
# Create Vector Database
# ---------------------------------------------------


def create_vector_embedding():

    if "vectors" not in st.session_state:

        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        loader = PyPDFDirectoryLoader("research_papers")

        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        documents = splitter.split_documents(docs)

        vectors = FAISS.from_documents(
            documents,
            embeddings,
        )

        st.session_state.embeddings = embeddings
        st.session_state.docs = docs
        st.session_state.documents = documents
        st.session_state.vectors = vectors


# ---------------------------------------------------
# Button
# ---------------------------------------------------

if st.button("Create Vector Database"):
    create_vector_embedding()
    st.success("Vector Database Created Successfully!")

# ---------------------------------------------------
# User Question
# ---------------------------------------------------

user_prompt = st.text_input("Ask a question from the research papers")

if user_prompt:

    if "vectors" not in st.session_state:
        st.warning("Please create the vector database first.")
        st.stop()

    retriever = st.session_state.vectors.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    document_chain = create_stuff_documents_chain(
        llm,
        prompt,
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain,
    )

    start = time.perf_counter()

    response = retrieval_chain.invoke(
        {
            "input": user_prompt,
        }
    )

    end = time.perf_counter()

    st.subheader("Answer")

    st.write(response["answer"])

    st.caption(f"Response Time: {end - start:.2f} seconds")

    with st.expander("Retrieved Chunks"):

        for i, doc in enumerate(response["context"], 1):

            st.markdown(f"### Chunk {i}")

            st.write(doc.page_content)

            st.divider()