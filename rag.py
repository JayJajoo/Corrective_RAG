from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()


class RAG():
    def __init__(self,chunk_size:int=1000, chunk_overlap:int=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = None
        self.retriever = None
        self.documents = []
        self.docs_summaries = []
    def load_urls(self,urls:list[str]):
        self.documents.extend(WebBaseLoader(urls).load())
    def load_texts(self,file_paths:list[str]):
        for path in file_paths:
            self.documents.extend(TextLoader(path).load())
    def load_docs(self,docs:list[Document]):
            self.documents.extend(docs)
    def generate_summary(self):
        llm = ChatOpenAI(model="gpt-4.1-nano",temperature=0.7)
        for doc in self.documents:
            prompt = (
                f"Provide the summary in very detail for the text below given source and its content:\n"
                f"SOURCE: {doc.metadata['source']} - CONTENT:{doc.page_content}."
            )
            result = llm.invoke(prompt)
            self.docs_summaries.append(Document(
                    page_content=f"Summary of {doc.metadata["source"]} is {result.content}",
                    metadata={'source': f"Summary of - {doc.metadata["source"]}"}
                )
            )
    def vectorize_documents(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        texts = text_splitter.split_documents(self.documents+self.docs_summaries)
        self.vectorstore = Chroma.from_documents(texts, self.embeddings,persist_directory="D:/LLM projects/corrective_rag/corrective_rag_db")
    def initialize_retriever(self):
        self.retriever = self.vectorstore.as_retriever(search_type="mmr",search_kwargs={'k': 6})
    def get_relevant_documents(self,query:str):
        docs = self.retriever.get_relevant_documents(query)
        return docs
    
