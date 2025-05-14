from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
import shutil
import streamlit as st

load_dotenv()

class RAG:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = None
        self.retriever = None
        self.documents = []

    def initialize_vector_store(self):
        BASE_PATH = os.path.dirname(os.path.abspath(__file__))
        self.vectorstore = Chroma(
            collection_name = "corrective_rag_db",
            persist_directory = f"{BASE_PATH}/db",
            embedding_function = self.embeddings
        )
    
    def get_existing_sources(self):
        data = self.vectorstore.get(include=["metadatas"])
        return set(meta["source"] for meta in data["metadatas"] if "source" in meta)
    
    def get_differences(self,new_docs):
        new_docs_sources = set(list(dict.fromkeys(doc.metadata["source"] for doc in new_docs)))
        existing_docs_sources = self.get_existing_sources()
        # print("New Docs :- ",new_docs_sources,"\n")
        # print("Existing source:",existing_docs_sources,"\n")
        sources_to_add = new_docs_sources - existing_docs_sources
        sources_to_del = existing_docs_sources - new_docs_sources
        return sources_to_add,sources_to_del
    
    def get_summary(self,doc):
        llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)
        prompt = (
            f"Provide the summary in very detail for the text below given source and its content:\n"
            f"SOURCE: {doc.metadata['source']} - CONTENT:{doc.page_content}."
        )
        result = llm.invoke(prompt)
        return Document(
            page_content=f"Summary of {doc.metadata['source']} is {result.content}",
            metadata={"source": doc.metadata['source']}
        )
    
    def sync_vectore_store(self,new_docs):
        sources_to_add,sources_to_delete = self.get_differences(new_docs)
        documents_to_add = [doc for doc in new_docs if doc.metadata["source"] in sources_to_add]
        summary_for_docs_to_add = [self.get_summary(doc) for doc in documents_to_add]
        # print("Sources to add\n",sources_to_add,"\nSources to delete:\n",sources_to_delete)
        if len(documents_to_add)>0:
            self.vectorstore.add_documents(documents_to_add+summary_for_docs_to_add)
        if len(sources_to_delete)>0:
            self.vectorstore.delete(
                where={"source":{"$in": list(sources_to_delete)}}
        )
        
        self.documents = (
            [doc for doc in self.documents if doc.metadata["source"] not in sources_to_delete] +
            documents_to_add +
            summary_for_docs_to_add
        )

    def initialize_retriever(self):
        self.retriever = self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6})

    def get_relevant_documents(self, query: str):
        return self.retriever.get_relevant_documents(query)