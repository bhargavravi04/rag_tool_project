from app import config
from langchain.chat_models import ChatOpenAI # type: ignore
from langchain.chains import RetrievalQA # type: ignore
from app.utils import load_catalog, chunk_text, create_vectorstore


def prepare_data():
    df = load_catalog()
    catalog_texts = df.apply(lambda row: ' '.join(map(str, row.values)), axis=1)
    all_chunks = []
    for text in catalog_texts:
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
    vectorstore = create_vectorstore(all_chunks)
    return vectorstore


def get_qa_chain(vectorstore):
    llm = ChatOpenAI(model_name=config.LLM_MODEL)
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return chain