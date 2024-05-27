from langchain_community.document_loaders import DirectoryLoader , PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import RetrievalQA
import os

from flask import Flask, render_template, request


os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")


FAISS_PATH = "/faiss"
llm = ChatOpenAI(model="gpt-3.5-turbo")

app = Flask(__name__)

def get_document_loader():
    loader = DirectoryLoader('static', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs

def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def get_embeddings():
    documents = get_document_loader()
    chunks = get_text_chunks(documents)
    db = FAISS.from_documents(
        chunks, OpenAIEmbeddings()
    )
    
    return db

def get_retriever():
    db = get_embeddings()
    retriever = db.as_retriever()
    return retriever


def process_llm_response(chain, question):

    llm_response = chain(question)
    
    print('Sources:')
    for i, source in enumerate(llm_response['source_documents']):
        result = llm_response['result']
        print(len(llm_response['result']))
        source_document = source.metadata['source']
        page_number = source.metadata['page']
        print(f"page {page_number}")
        source_document = source_document[7:]
        
        return result, source_document, page_number

def get_chain():
    retriever = get_retriever()
        
    chain = RetrievalQA.from_chain_type(llm = llm,
                                        chain_type="stuff",
                                        retriever = retriever,
                                        return_source_documents = True
                                        )
    return chain
    
@app.route('/')
def index():
    return render_template('index.html')

    
@app.route('/', methods = ['GET', 'POST'])
def document_display():
    retriever = get_retriever()
    
    chain = RetrievalQA.from_chain_type(llm = llm,
                                        chain_type="stuff",
                                        retriever = retriever,
                                        return_source_documents = True
                                        )
    
    question = request.form['question']
    try:
        result, source_document, page_number = process_llm_response(chain=chain, question=question)
        page_number = page_number + 1
        source_document =  os.path.join("static",source_document, f"#page={page_number}" )
    except Exception as e:
        print(f'error - {e}')
        result, source_document, page_number = "", "", ""
    print(source_document)
    return render_template('index.html', result = result, source_document=source_document, page_number=page_number)


if __name__ == "__main__":
    app.run(debug=True)