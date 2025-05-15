import operator
from pydantic import BaseModel,Field
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_community.tools import TavilySearchResults
from typing import TypedDict,Annotated,Sequence,Literal
from langchain.schema import Document
from langchain_core.messages import BaseMessage,AIMessage,ToolMessage
from dotenv import load_dotenv
from rag import RAG
from langchain_community.document_loaders import WebBaseLoader


load_dotenv()

rag = None

def initalize_rag(urls, docs):
    global rag  
    
    if rag is None:
        rag = RAG()
    
    data = []

    if urls:
        data.extend(WebBaseLoader(urls).load())
    if docs:
        data.extend(docs)
    
    if len(data)>0:
        rag.initialize_vector_store()
        rag.sync_vectore_store(data)
        rag.initialize_retriever()

class DocumentRelevancy2(BaseModel):
    relevant: list[str] = Field(description="List of response suggestiong wether documents are relevant or not. Example ['yes','no','yes'...]") 

class GeneralMessage(BaseModel):
    isQuestion: str = Field(description="Wether user input is really a query or not") 

class AgentState(TypedDict):
    """State of the agent."""
    query: str
    isQuestion:str
    web_search_results: list[Document]
    documents: list[Document]
    requires_web_search:bool
    messages: Annotated[Sequence[BaseMessage],operator.add]
    answer:str

def capture_intent(state:AgentState):
    
    query = state["query"]

    sys_msg = """You are an intelligent assistant. Your job is to analyze a user’s message and decide whether it is:
    Just a greeting or casual message (like “hi”, “hello”, “thanks”, “how are you?”, “good morning”), which does not require further processing, or
    A meaningful question or request that likely requires retrieval, summarization, or external web search.
    Rules:
    If the message is only a greeting, thank-you note, small talk, or unrelated pleasantry, respond with: no
    If the message asks for information, makes a request, or has keywords suggesting a topic, respond with: yes
    Output Format:
    Respond only with a single word: yes or no (lowercase, no punctuation)
    Examples:
    “hi there” → no
    “thanks!” → no
    “can you summarize this article for me?” → yes
    “what is LangGraph?” → yes
    “hello, how’s it going?” → no
    “hey, what’s the latest news on LLMs?” → yes"""

    chat_template = ChatPromptTemplate.from_messages([
        ("system",sys_msg),
        ("user",f"Judge the below QUERY as instructed: \n{query}")
    ])

    llm = ChatOpenAI(model="gpt-4.1-nano").with_structured_output(GeneralMessage)

    prompt = chat_template.format_prompt(query=query)

    result = llm.invoke(prompt)

    return {"isQuestion":result.isQuestion}

def initial_route(State:AgentState):
    isQuestion = State["isQuestion"]
    if isQuestion=="yes":
        return "rephrase_query"
    return "summarize"

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


class DocumentRelevancy(BaseModel):
    relevant: str = Field(description="Wether document is relevant or not.") 

def quality_grader2(state: AgentState):
    """Grade the quality of each retrieved document individually for relevance."""

    if len(state["documents"]) == 0:
        return {"documents": [], "requires_web_search": True}

    documents = state["documents"]
    query = state["query"]
    relevance_results = []
    new_docs_list = []

    sys_msg = """
    You are an assistant tasked with determining whether a document is at least **partially relevant** to the given query.

    ### Query:
    "{query}"

    ### Instructions:
    - If the query explicitly requests or implies a web search, the document provided might not be directly relevant — still evaluate it carefully.
    - If the document is very long, summarize it first before assessing relevance.
    - Review the document **as if it were written in the context of the query**.
    - If the document includes information, phrases, or implications that could support, explain, or relate to the query, even indirectly, respond with "yes".
    - If the document contains entirely unrelated information, respond with "no".
    - Consider implied connections — for example, an increase in profit could be relevant to a query about stock market growth.

    ### Output format:
    Respond with a single string: "yes" or "no".
    """

    chat_template = ChatPromptTemplate.from_messages([
        ("system", sys_msg),
        ("user", "DOCUMENT:\n{document}")
    ])

    llm = ChatOpenAI(model="gpt-4.1-nano").with_structured_output(str)

    for doc in documents:
        prompt = chat_template.format_prompt(document=doc, query=query)
        result = llm.invoke(prompt).strip().lower()
        relevance_results.append(result)
        if result == "yes":
            new_docs_list.append(doc)

    # Apply 20% threshold condition
    if len(new_docs_list) == 0 or len(new_docs_list) / len(documents) < 0.30:
        return {"documents": new_docs_list, "requires_web_search": True}

    return {"documents": new_docs_list, "requires_web_search": False}


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
    1. Do not generate random information or facts that except the things provided to you.
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


# def quality_grader2(state:AgentState):
#     """Grade the quality of the retrieved documents."""
#     if len(state["documents"]) > 0:
#         documents = state["documents"]
#         query = state["query"]
        
#         sys_msg = """
#         You are an assistant tasked with determining whether each document is at least **partially relevant** to the given query.

#         ### Query:
#         "{query}"

#         ### Instructions:
#         - If the query explicitly requests or implies a web search, the documents provided might not be directly relevant — still evaluate them carefully.
#         - If a document is very long, summarize it first before assessing relevance.
#         - Review each document **as if it were written in the context of the query**.
#         - If a document includes information, phrases, or implications that could support, explain, or relate to the query, even indirectly, respond with "yes".
#         - If a document contains entirely unrelated information, respond with "no".
#         - Consider implied connections — for example, an increase in profit could be relevant to a query about stock market growth.
#         - The number of items in your output list must exactly match the number of documents provided.

#         ### Output format:
#         Return a list of responses corresponding to the documents’ order: ["yes", "no", "yes"]
#         """

#         chat_template = ChatPromptTemplate.from_messages([
#             ("system", sys_msg),
#             ("user", "NUMBER OF DOCUMENTS : {doc_len}\n\nDOCUMENTS : {documents}"),
#         ])

#         llm = ChatOpenAI(model="gpt-4.1-nano").with_structured_output(DocumentRelevancy2)
#         prompt = chat_template.format_prompt(documents=str(documents),doc_len=len(documents),query=query)
#         result = llm.invoke(prompt)
        
#         new_docs_list = []
        
#         for idx,relevance in enumerate(result.relevant):
#             if relevance=="yes":
#                 new_docs_list.append(documents[idx])
        
#         if len(new_docs_list) == 0 or len(new_docs_list)/len(documents) < 0.20:
#             return {"documents": new_docs_list,"requires_web_search":True}
    
#     return {"documents": state["documents"],"requires_web_search":False}

