import pandas as pd # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain.embeddings.openai import OpenAIEmbeddings # type: ignore
from langchain.vectorstores import FAISS # type: ignore
from app import config


def load_catalog(file_path="product_catalog/catalog.csv"):
    df = pd.read_csv(file_path)
    return df

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    return splitter.split_text(text)

def create_vectorstore(docs):
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    return FAISS.from_texts(docs, embedding=embeddings)