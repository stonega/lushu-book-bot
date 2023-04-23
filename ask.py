from langchain import FAISS, OpenAI, PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()
db = FAISS.load_local('faiss_index', embeddings)

qa_chain = load_qa_with_sources_chain(OpenAI(
    temperature=0.2), chain_type="map_reduce")

chain = RetrievalQAWithSourcesChain(
    combine_documents_chain=qa_chain, retriever=db.as_retriever())


def ask_question(question):
    result = chain({"question": question}, return_only_outputs=True)
    return result
