import validators,streamlit as st
from langchain_classic.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain


##Streamlit APP
st.set_page_config(page_title="langchain summerizer",page_icon="🦜")
st.title("🦜 LangChain summarizer text from YT and Website.")

with st.sidebar:
    groq_api_key = st.text_input("GROQ API KEY",type="password")

genric_url = st.text_input("URL",label_visibility="collapsed")

if groq_api_key:
    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama-3.3-70b-versatile"   # or another supported model
    )
else:
    st.warning("Please enter your Groq API key.")
    st.stop()
prompt_template = """
    provide the in detail summary for the following content in 300 words.
    Content:{text}
    """
prompt = PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("summarize the content from yt or website"):
    if not groq_api_key.strip() or not genric_url.strip():
        st.error("please provide the information to get get staretd")
    elif not validators.url(genric_url):
        st.error("please enter a valid url it can may be a yt or website url")
    else:
        try:
            with st.spinner("waiting..."):
                if "youtube.com" in genric_url:
                    loader = YoutubeLoader.from_youtube_url(genric_url,language=["en-IN", "en", "hi"],add_video_info=False)
                else:
                    loader = UnstructuredURLLoader(urls=[genric_url],ssl_verify=False,
                                                   headers={"User-Agent": (
                                                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                                                    "Chrome/138.0.0.0 Safari/537.36"
                                                    ),
                                                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                                                    "Accept-Language": "en-US,en;q=0.9",
                                                    "Referer": "https://www.google.com/",
                                                    "Connection": "keep-alive",})
                docs = loader.load()
                
                # chain for Summarization
                chain = load_summarize_chain(llm=llm,chain_type="stuff",prompt=prompt)
                output_summary = chain.run(docs)
                st.success(output_summary)
                
        except Exception as e:
            st.exception(f"Exception:{e}")
            
