import os
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from rag import RAG
from agent import AgentState, rephrase_query, retriver, quality_grader2, router, web_search, summarize, initalize_rag, capture_intent,initial_route
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
from langchain.schema import Document
import pandas as pd
import docx2txt
import PyPDF2
import time
import os
import tomllib 

with open("secrets.toml", "rb") as f:
    config = tomllib.load(f)  

os.environ["LANGSMITH_TRACING"] = str(config["langsmith"]["tracing"])
os.environ["LANGSMITH_ENDPOINT"] = config["langsmith"]["endpoint"]
os.environ["LANGSMITH_API_KEY"] = config["langsmith"]["api_key"]
os.environ["LANGSMITH_PROJECT"] = config["langsmith"]["project"]
os.environ["OPENAI_API_KEY"] = config["openai"]["api_key"]
os.environ["TAVILY_API_KEY"] = config["tavily"]["api_key"]


def initialize_memory():
    memory = MemorySaver()
    return memory

def initialize_agent(docs=None, urls=None):
    initalize_rag(urls=urls, docs=docs)

    workflow = StateGraph(AgentState)
    workflow.add_node("capture_intent",capture_intent)
    workflow.add_node("initial_route",initial_route)
    workflow.add_node("rephrase_query", rephrase_query)
    workflow.add_node("retriver", retriver)
    workflow.add_node("quality_grader", quality_grader2)
    workflow.add_node("router", router)
    workflow.add_node("web_search", web_search)
    workflow.add_node("summarize", summarize)

    workflow.add_edge(START,"capture_intent")
    workflow.add_conditional_edges("capture_intent", initial_route, {"rephrase_query": "rephrase_query", "summarize": "summarize"})
    workflow.add_edge("rephrase_query", "retriver")
    workflow.add_edge("retriver", "quality_grader")
    workflow.add_conditional_edges("quality_grader", router, {"web_search": "web_search", "summarize": "summarize"})
    workflow.add_edge("web_search", "summarize")
    workflow.add_edge("summarize", END)

    agent = workflow.compile(checkpointer=st.session_state.memory)
    return agent

def main():
    if "disabled" not in st.session_state:
        st.session_state.disabled = True
    if "config" not in st.session_state:
        st.session_state.config = {"configurable": {"thread_id": "1"}}
    if "memory" not in st.session_state:
        st.session_state.memory = initialize_memory()
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "docs" not in st.session_state:
        st.session_state.docs = []
    if "urls" not in st.session_state:
        st.session_state.urls = []

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload text files", type=["txt", "docx", "pdf", "csv"], accept_multiple_files=True)
        links = st.text_area(label="Paste your links here separated by ','")
        button = st.button(label="Upload Content")

        if button:
            start = time.time()
            with st.spinner(text="Loading..."):
                docs = []
                urls = []

                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        file_type = uploaded_file.name.split(".")[-1].lower()
                        content = ""
                        if file_type == "txt":
                            content = uploaded_file.getvalue().decode("utf-8")
                        elif file_type == "pdf":
                            pdf_reader = PyPDF2.PdfReader(uploaded_file)
                            content = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
                        elif file_type == "docx":
                            content = docx2txt.process(uploaded_file)
                        elif file_type == "csv":
                            df = pd.read_csv(uploaded_file)
                            content = df.to_csv(index=False)

                        if content:
                            docs.append(Document(page_content=content, metadata={"source": uploaded_file.name}))

                if links:
                    urls = [url.strip() for url in links.split(",") if url.strip()]

                if docs or urls:
                    st.session_state.docs = docs
                    st.session_state.urls = urls
                    st.session_state.agent = initialize_agent(docs=docs, urls=urls)
                    st.session_state.disabled = False
                    end = time.time()
                    st.success(f"Vector Store Built Successfully in {end-start:.2f} seconds !")
                else:
                    st.warning("No valid content uploaded or entered.")


    chat_input = st.chat_input(
        placeholder="Please upload some documents to start.",
        disabled=st.session_state.disabled
    )

    if chat_input and st.session_state.agent is not None:
        state = {
            "query": str(chat_input),
            "rephrased_query": None,
            "web_search_results": [],
            "documents": [],
            "requires_web_search": None,
            "messages": [HumanMessage(content=chat_input)],
            "answer": None,
        }
        result = st.session_state.agent.invoke(state, config=st.session_state.config)
        st.session_state.messages = result["messages"]

    for msg in st.session_state.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)


if __name__ == "__main__":
    main()
