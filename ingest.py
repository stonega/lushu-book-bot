"""This is the logic for ingesting Notion data into LangChain."""
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import GitLoader
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from dotenv import load_dotenv

load_dotenv()

loader = GitLoader(
    clone_url="https://github.com/stonega/lushu-book",
    repo_path="./lushu-book/",
    branch="main",
    file_filter=lambda file_path: file_path.endswith(".md")
)

documents = loader.load()

# Here we split the documents, as needed, into smaller chunks.
# We do this due to the context limits of the LLMs.
text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
docs = text_splitter.split_documents(documents)

# docs = []
# metadatas = []
# for i, d in enumerate(data):
#     splits = text_splitter.split_text(d)
#     docs.extend(splits)
#     metadatas.extend([{"source": sources[i]}] * len(splits))


# Here we create a vector store from the documents and save it to disk.
store = FAISS.from_documents(docs, OpenAIEmbeddings(), metadatas=metadatas)
faiss.write_index(store.index, "docs.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)