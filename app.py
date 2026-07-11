from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

#langsmith tracking

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "simple QNA Chatbot"


#prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful assistent to respond user question in a poliet way"),
        ("user","Question:{question}")
    ]
)


def generate_response(question,api_key,llm,temperature,max_tokens):
    grog_api_key = api_key
    llm = ChatGroq(
        model=llm,
        groq_api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )
    #output parser
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer

##title of the app

st.title("Enhanced Q&A chatbot with Groq")

st.sidebar.title("settings")
api_key = st.sidebar.text_input(
    "Enter your Groq API Key",
    type="password"
)
#drop down to select various groq models

llm = st.sidebar.selectbox("select the llm model: ",["openai/gpt-oss-20b","llama-3.1-8b-instant","openai/gpt-oss-120b"])

tempareture = st.sidebar.slider("Tempareture",min_value=0.01,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

#main_interface
user_input = st.text_input("you: ")

if user_input:
    response = generate_response(user_input,api_key,llm,tempareture,max_tokens)
    st.write(response)

else:
    st.write("enter your query iam here to respond")


