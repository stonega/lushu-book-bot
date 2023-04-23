from pathlib import Path
from langchain.document_loaders import SRTLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

ps = list(Path("lushu-book/").glob("**/*.mdx"))

docs = []
for p in ps:
    loader = SRTLoader(p)
    docs.extend(loader.load())

# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(docs)

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size = 1000, chunk_overlap = 100)
texts=text_splitter.split_documents(docs)

embeddings=OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)
# db.save(dataset_path)
db.save_local("faiss_index")
