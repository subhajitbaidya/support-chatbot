
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.cleantext import clean_text


def load_documents():
    loader = PyPDFDirectoryLoader("./data/pdf")
    docs = loader.load()
    return [
        Document(
            page_content=clean_text(doc.page_content),
            metadata=doc.metadata
        )
        for doc in docs
    ]


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


def build_vectorstore():
    docs = load_documents()
    splits = split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vstore = InMemoryVectorStore(embeddings)
    vstore.add_documents(splits)

    return vstore


if __name__ == "__main__":
    print(build_vectorstore())
