import operator
from pydantic import BaseModel,Field
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults,TavilySearchResults
from typing import TypedDict,Annotated,Sequence,Literal
from langchain_core.messages import BaseMessage,AIMessage,ToolMessage
from dotenv import load_dotenv
from rag import RAG
load_dotenv()

rag = None

def initalize_rag(urls,file_paths,docs):
    global rag
    rag = RAG()
    if urls:
        rag.load_urls(urls=urls)
    if docs:
        rag.load_docs(docs=docs)
    if file_paths:
        rag.load_texts(file_paths=file_paths)
    rag.vectorize_documents()
    rag.initialize_retriever()

class DocumentRelevancy(BaseModel):
    relevant: str = Field(description="Wether document is relevant or not.") 

class AgentState(TypedDict):
    """State of the agent."""
    query: str
    rephrased_query:str
    web_search_results: list[Document]
    documents: list[Document]
    requires_web_search:bool
    messages: Annotated[Sequence[BaseMessage],operator.add]
    answer:str

def rephrase_query(state:AgentState):
    """Checks if the query is related to previous chats."""
    if (len(state["messages"])>1):
        sys_msg = """
        Given the chat history and a new user question, determine whether the question depends on the previous conversation.
        If it does, rephrase it into a complete, natural, and contextually grounded user question that could be clearly understood by a language model without needing prior context.
        If the question is already self-contained and unambiguous, return it unchanged.

        Important guidelines:
        - Your output must always be a full user-style question — not a meta-comment or assistant-side interpretation.
        - Do NOT invent or assume facts beyond what’s explicitly present in the conversation.
        - Only introduce prior context when it’s needed to make the question understandable in isolation.
        - Be concise but complete — include only the necessary references.

        Return ONLY the rewritten or original user question — no extra commentary or formatting.
        CONTEXT: {context}
        """


        chat_template = ChatPromptTemplate.from_messages([
            ("system", sys_msg),
            ("user", "Please rephrase the following query if required: {query}"),
        ])
        query = state["query"]
        messages = [msg for msg in state["messages"] if isinstance(msg,ToolMessage)==False] 
        llm = ChatOpenAI(model="gpt-4.1-nano")
        prompt = chat_template.format_prompt(query=query, context=messages)
        response = llm.invoke(prompt)
        return {"query":response.content}
    return {"query":state["query"]}

def retriver(state:AgentState):
    """Retrieve relevant documents from the vectorstore."""
    global rag
    query = state["query"]
    docs = rag.get_relevant_documents(query)
    return {"documents": docs}

def quality_grader(state:AgentState):
    """Grade the quality of the retrieved documents."""
    if len(state["documents"]) > 0:
    
        documents = state["documents"]
        query = state["query"]

        sys_msg = """
        Your task is to assess whether the provided document is paritally relevant to the given query.

        Response Guidelines:
        - If the document is paritally relevant to the query, respond with "yes".
        - If the document is not even paritally relevant, respond with "no".

        Additional Instructions:
        Be aware that the text may contain isolated words or short phrases that lack surrounding context. These may still represent meaningful entities or concepts. Use your general knowledge and reasoning to interpret such cases and assess relevance, even when explicit relationships are not stated.
        """


        chat_template = ChatPromptTemplate.from_messages([
            ("system", sys_msg),
            ("user", "DOCUMENT: {document}\nQUERY: {query}"),
        ])

        llm = ChatOpenAI(model="gpt-4.1-nano",temperature=0.9).with_structured_output(DocumentRelevancy)
        new_docs_list = []

        for doc in documents:
            prompt = chat_template.format_prompt(document=doc.page_content,query=query)
            response = llm.invoke(prompt)
            if str(response.relevant).endswith("yes"):
                new_docs_list.append(doc)    
        # print(f"Relevant Chunks/Total Chunks = {len(new_docs_list)}/{len(documents)}")
        if len(new_docs_list) == 0 or len(new_docs_list)/len(documents) < 0.10:
            return {"documents": new_docs_list,"requires_web_search":True}
    
    return {"documents": new_docs_list,"requires_web_search":False}

def router(state:AgentState):
    """Route the query to the appropriate function based on the state."""
    if state["requires_web_search"] == True:
        return "web_search"
    return "summarize"

def web_search(state:AgentState):
    """Perform a web search using DuckDuckGo."""
    query = state["query"]
    search = TavilySearchResults(max_results = 1)
    # search = DuckDuckGoSearchResults(num_results=1)
    results = search.invoke(query)
    return {"web_search_results": results,"requires_web_search":False}

def summarize(state:AgentState):
    query = state["query"]
    documents = state["documents"]
    web_search_results = state["web_search_results"]
    messages = state["messages"][-10:]
    messages = [msg for msg in state["messages"] if isinstance(msg,ToolMessage)==False] 


    llm= ChatOpenAI(model="gpt-4.1-nano",temperature=0.7)

    sys_msg = """You are given a user question, the chat history, and a list of documents providing context. Your task is to generate a concise and relevant summary that directly addresses the user’s question.
    Instructions:
    1. Use both the chat history and the documents to inform your summary.
    2. Prioritize the content and insights from the documents over the chat history.
    3. Keep the tone conversational and user-friendly.
    4. Ensure the summary is focused, informative, and directly aligned with the user's question."""

    chat_template = ChatPromptTemplate.from_messages([
        ("system", sys_msg),
        ("user", "CHAT HISTORY: {messages}\nDOCUMENTS: {documents}\nAnswer this QUERY: {query} uing the CHAT HISTORY and DOCUMENTS and foloow the innstructions provided."),
    ])
    
    prompt = chat_template.format_prompt(query=query, messages=messages, documents=documents+[web_search_results])
    response = llm.invoke(prompt)
    return {"answer": response.content,"messages":[AIMessage(content=response.content)]}






















# class GreetingMessages(BaseModel):
#     is_greeting: str = Field(description="Wether user input is greeting or not.") 

# class AgentState(TypedDict):
#     """State of the agent."""
#     query: str
#     query_is_greeting:str
#     web_search_results: list[Document]
#     documents: list[Document]
#     requires_web_search:bool
#     messages: Annotated[Sequence[BaseMessage],operator.add]
#     answer:str

# def is_greeting(state:AgentState):
    
#     query = state["query"]

#     sys_msg = "You are an assistant. Your task is to determine whether the given query is purely a greeting or opening message, such as 'hello', " \
#     "'hi', 'how are you', 'thank you', etc. If the message is only a greeting, return 'yes'. If it includes any additional content, such as a " \
#     "question or a specific request, return 'no'."

#     chat_template = ChatPromptTemplate.from_messages([
#         ("system",sys_msg),
#         ("user",f"Here's the query:\n{query}")
#     ])

#     llm = ChatOpenAI(model="gpt-4.1-nano").with_structured_output(GreetingMessages)

#     prompt = chat_template.format_prompt(query=query)

#     result = llm.invoke(prompt)

#     return {"query_is_greeting":result.is_greeting}

