from langchain import FAISS, OpenAI, PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
import argparse
from dotenv import load_dotenv

def ask_question(question):

    load_dotenv()

    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local('faiss_index', embeddings)
    
    # question_prompt_template = """Use the following pieces of context to answer the question at the end, 请必须使用中文回答。
    # {context}
    # Question: {question}
    # :"""
    # QUESTION_PROMPT = PromptTemplate(
    #     template=question_prompt_template, input_variables=["context", "question"]
    # )

    qa_chain = load_qa_with_sources_chain(OpenAI(
        temperature=0.2), chain_type="map_reduce")

    chain = RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain, retriever=db.as_retriever())

    result = chain({"question": question}, return_only_outputs=True)
    return result


parser = argparse.ArgumentParser(
    description='Ask a question to the Lushu Book.')
parser.add_argument('question', type=str,
                    help='The question to ask the Lushu Book')
args = parser.parse_args()

result = ask_question(args.question)

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
