from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_ollama.chat_models import ChatOllama
from langchain.tools import tool
from langsmith import traceable
from dotenv import load_dotenv
from utils.cleantext import clean_text
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore


load_dotenv(dotenv_path=".env", override=True)


class CustomerSupportAgent:
    def __init__(self, task):
        self.model = ChatOllama(
            model="llama3.2:latest",
            temperature=0.5,
            streaming=True
        )
        self.task = task

    def model(self):
        pass

    @traceable(name="embedding_model")
    def __embedding_model(self):
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2")

    @traceable(name="load_documents")
    def __directoryloader(self):
        loader = PyPDFDirectoryLoader('./data/pdf')
        docs = loader.load()
        cleaned_docs = [
            Document(
                page_content=clean_text(doc.page_content),
                metadata=doc.metadata
            )
            for doc in docs
        ]

        return cleaned_docs

    def __textsplitter(self):
        docs = self.__directoryloader()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            add_start_index=True,
        )

        all_splits = splitter.split_documents(docs)
        return all_splits

    def vectorstore(self):
        vector_store = InMemoryVectorStore(self.__embedding_model())
        docs = self.__textsplitter()
        doc_id = vector_store.add_documents(documents=docs)
        return doc_id


if __name__ == "__main__":
    agent = CustomerSupportAgent("customer support")
    print(len(agent.textsplitter()))
