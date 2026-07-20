import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_classic.agents import AgentExecutor,create_react_agent
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.tools import tool
import numexpr


#set the page config

st.set_page_config(page_title="MathGPT",page_icon="➕")
st.title("Text to math problem solver using Google Gemma")

groq_api_key = st.sidebar.text_input(label="Groq API key",type="password")

if not groq_api_key:
    st.info("please provide groq api key to continue")
    st.stop()
llm = llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    api_key=groq_api_key,
    temperature=0
    )

#initializing tools
wiki = WikipediaAPIWrapper()

@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for information."""
    return wiki.run(query)

prompt =  PromptTemplate.from_template("""
You are a helpful AI assistant.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question
Thought: think about what to do
Action: one of [{tool_names}]
Action Input: input to the action
Observation: result of the action
...
Thought: I now know the final answer
Final Answer: the answer

Question: {input}

Thought: {agent_scratchpad}
""")

reasoning_chain = (
    prompt
    | llm
    | StrOutputParser()
)
@tool
def reasoning(question: str) -> str:
    """Useful for logical reasoning questions."""
    return reasoning_chain.invoke({"question": question})

@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions."""

    return str(numexpr.evaluate(expression))

tools = [
    wikipedia_search,
    calculator,
    reasoning
]

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

question = st.text_area("Enter your Question:","")
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm your AI assistant. How can I help you today?"
        }
    ]
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.button("Find My Answer"):
    if question.strip():

        # Display user message
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )
        st.chat_message("user").write(question)

        with st.spinner("Generating response..."):
            try:
                # Streamlit callback handler
                st_cb = StreamlitCallbackHandler(
                    st.container(),
                    expand_new_thoughts=False
                )

                # Invoke the agent
                response = agent_executor.invoke(
                    {"input": question},
                    config={
                        "callbacks": [st_cb]
                    }
                )

                answer = response["output"]

                # Save assistant response
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

                # Display assistant response
                st.chat_message("assistant").write(answer)

            except Exception as e:
                st.error(f"Error: {e}")

    else:
        st.warning("Please enter a question.")