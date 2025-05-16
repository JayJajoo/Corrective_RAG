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

class DocumentRelevancy(BaseModel):
    relevant: str = Field(description="Wether document is relevant or not.") 

class WebSearchReuired(BaseModel):
    isAskingForWebSearch: str = Field(description="Wether user is asking for web search or not.") 

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
        ("user","Judge the below QUERY as instructed: \n{query}")
    ])

    llm = ChatOpenAI(model="gpt-4.1-nano").with_structured_output(GeneralMessage)

    prompt = chat_template.format_prompt(query=str(query))

    result = llm.invoke(prompt)

    return {"isQuestion":result.isQuestion}

def initial_route(State:AgentState):
    isQuestion = State["isQuestion"]
    if isQuestion=="yes":
        return "is_asking_for_web_search"
    return "summarize"

def rephrase_query(state:AgentState):
    """Checks if the query is related to previous chats."""
    if (len(state["messages"])>1):
        sys_msg = """
        You are a helpful assistant. Given the chat history and a new user question, determine whether the question relies on prior context. 
        If it does, rephrase it into a complete, natural, and contextually grounded question that is understandable on its own. 
        If the question is already clear and self-contained, return it unchanged.

        EXAMPLE 1 :
        CONTEXT:
        User: Who is Narendra Modi?
        NEW USER QUESTION: What is his age?
        Rephrased question: What is the age of Narendra Modi?

        EXAMPLE 2 :
        CONTEXT:
        User: What is travelling salesman problem?
        NEW USER QUESTION: give me code for that?
        Rephrased question: What is the code for travelling salesman problem?

        EXAMPLE 2 :
        CONTEXT:
        User: where is california?
        NEW USER QUESTION: how's its weather?
        Rephrased question: how's the weather in california?
        
        Guidelines:
        - Always return a complete user-style question — never commentary or explanations.
        - Do NOT invent or assume any facts that aren't explicitly mentioned in the conversation.
        - Only include prior context if it's necessary to make the question understandable in isolation.
        - Keep it concise but complete — include only what's needed.

        Output ONLY the rephrased or original question — no formatting, no commentary.

        CONTEXT: {context}
        """

        chat_template = ChatPromptTemplate.from_messages([
            ("system", sys_msg),
            ("user", "Please rephrase the following query if required: {query}"),
        ])
        query = state["query"]
        messages = [str(msg.content) for msg in state["messages"] if isinstance(msg,ToolMessage)==False] 
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

def quality_grader2(state: AgentState):
    """Grade the quality of each retrieved document individually for relevance."""

    if len(state["documents"]) == 0:
        return {"documents": [], "requires_web_search": True}

    documents = state["documents"]
    query = state["query"]
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

    llm = ChatOpenAI(model="gpt-4.1-nano").with_structured_output(DocumentRelevancy)

    for doc in documents:
        prompt = chat_template.format_prompt(document=doc, query=query)
        result = llm.invoke(prompt)
        if result.relevant == "yes":
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

def summarize(state: AgentState):
    query = state["query"]
    documents = state["documents"]
    web_search_results = state["web_search_results"]
    messages = state["messages"][-10:]
    messages = [msg for msg in state["messages"] if not isinstance(msg, ToolMessage)] 

    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)

    sys_msg = """You are given a user question, the chat history, and a list of documents providing context. 
    Additionally, you may receive web search results as part of the documents.
    Your task is to generate a concise and relevant summary that directly addresses the user’s question.

    Instructions:
    1. Do NOT say that you cannot perform a web search. Assume web search results are provided as part of the input.
    2. Use both the chat history, documents, and web search results to inform your summary.
    3. Prioritize the content and insights from documents and web search results over the chat history.
    4. Keep your tone conversational, friendly, and natural.
    5. Ensure the summary is focused, informative, and directly aligned with the user's question.
    6. Do NOT generate any information beyond what is provided to you.
    """

    chat_template = ChatPromptTemplate.from_messages([
        ("system", sys_msg),
        ("user", "CHAT HISTORY: {messages}\nDOCUMENTS AND WEB SEARCH RESULTS: {documents}\nAnswer this QUERY: {query} using the CHAT HISTORY and DOCUMENTS and follow the instructions provided."),
    ])

    # Combine documents with web_search_results (assuming web_search_results is a list or compatible)
    combined_docs = documents + (web_search_results if isinstance(web_search_results, list) else [web_search_results])

    prompt = chat_template.format_prompt(query=query, messages=messages, documents=combined_docs)
    response = llm.invoke(prompt)
    return {"answer": response.content, "messages": [AIMessage(content=response.content)]}

def is_asking_for_web_search(state:AgentState):
    query = state["query"]
    llm = ChatOpenAI(model="gpt-4.1-nano").with_structured_output(WebSearchReuired)
    sys_msg = (
        "You are a helpful assistant. Your task is to determine if the user is explicitly or implicitly requesting a web search. "
        "Respond with 'yes' if the user's query includes phrases like 'search the internet', 'search online', 'web search', "
        "'look up online resources', or any similar expressions that indicate searching the web. "
        "Respond with 'no' if the query refers to local document search, such as 'search documents', 'look up in docs', or 'look up in documents'."
    )
    template = ChatPromptTemplate.from_messages([
        ("system",sys_msg),
        ("user","Decide whether user is asking for web search or not for below QUERY:\n\n{query}")
    ])
    prompt = template.format_prompt(query=str(query))
    result = llm.invoke(prompt)
    if result.isAskingForWebSearch == "yes":
        return {"requires_web_search":True}
    return {"requires_web_search":False}

def rag_or_web_router(state:AgentState):
    if state["requires_web_search"]==True:
        return "web_search"
    return "call_rag"

# class DocumentRelevancy2(BaseModel):
#     relevant: list[str] = Field(description="List of response suggestiong wether documents are relevant or not. Example ['yes','no','yes'...]") 

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

