# RAG Q&A converstion with pdf inculiding chat history
import streamlit as st
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

##setup streamlit
st.title("Conversational RAG with PDF uploads and chat history")
st.write("Upload pdf and chat with there content")

api_key = st.text_input("Enter you Groq Api key",type="password")

#check if groq api key is provided

if api_key:
    llm = ChatGroq(
        model_name="openai/gpt-oss-20b",
        groq_api_key = api_key
    )

    session_id = st.text_input("Session ID",value="default_session")
    #statefully manage the chat history
    if 'store' not in st.session_state:
        st.session_state.store = {}
    
    upload_files = st.file_uploader("choose A pdf file",type="pdf",accept_multiple_files=True)

    #process uploaded pdf
    if upload_files:
        documents = []
        for upload_file in upload_files:
            temppdf = f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(upload_file.getvalue())
                file_name = upload_file.name

            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        #split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(splits,embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k":4})

        contexualize_q_system_prompt = (
            """
            Given the chat history and the latest user question,
            rewrite the latest question into a standalone question
            that can be understood without the chat history.

            Do NOT answer the question.

            If the question is already standalone,
            return it unchanged.
            """
        )
        contexualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system",contexualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human","{input}"),
                ]
        )
        history_aware_retriever = create_history_aware_retriever(llm,retriever,contexualize_q_prompt)

        #answer prompt

        system_prompt = (
            """Instructions:

            1. Use the retrieved context as your primary source of information.
            2. Use the chat history to understand follow-up questions and references such as:
            - "Explain that again."
            - "What about the second method?"
            - "Who created it?"
            3. Never invent or assume information that is not present in the retrieved context.
            4. If the answer cannot be found in the retrieved context, respond exactly:
            "I don't have enough information in the uploaded documents to answer that."
            5. If the question is unrelated to the uploaded documents, politely inform the user that you can only answer based on the uploaded PDFs.
            6. If multiple retrieved document chunks contain relevant information, combine them into one coherent answer.
            7. Preserve technical terms, equations, names, and numerical values exactly as they appear.
            8. If there is conflicting information in the retrieved context, mention the conflict instead of guessing.
            9. Format answers using Markdown with headings, bullet points, or numbered lists when appropriate.
            10. Be concise by default, but provide detailed explanations if requested."""
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session]=ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("what is your qestion: ")
        if user_input:
            session_history = get_session_history(session_id)

            response = conversational_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":session_id}
                },
            )
            st.write(st.session_state.store)
            st.write("Assistent:",response["answer"])
            st.write("Chat History:",session_history.messages)


else:
    st.warning("please provide your groq api key")
