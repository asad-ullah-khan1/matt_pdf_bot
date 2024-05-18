import streamlit as st
from PyPDF2 import PdfReader
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import cassio

# Setup (Do NOT expose sensitive keys in your code. Use environment variables or other secure methods.)
ASTRA_DB_APPLICATION_TOKEN = st.secrets["your_astra_db_application_token"]
ASTRA_DB_ID = st.secrets["your_astra_db_id"]
OPENAI_API_KEY = st.secrets["your_openai_api_key"]

# Initialize the system
def initialize_system():
    cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    astra_vector_store = Cassandra(
        embedding=embedding,
        table_name="qa_vector_store",
        session=None,
        keyspace=None,
    )
    astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
    return llm, embedding, astra_vector_store, astra_vector_index

# Read text from a PDF
def process_pdf(pdf_file):
    pdfreader = PdfReader(pdf_file)
    raw_text = ''
    for page in pdfreader.pages:
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

# Split the text and add it to the vector store
def prepare_text_vector_store(astra_vector_store, raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    astra_vector_store.add_texts(texts[:50])

# Query the PDF content
def query_pdf(astra_vector_index, llm, query_text):
    answer = astra_vector_index.query(query_text, llm=llm).strip()
    return answer

# Streamlit app starts here
st.title('PDF Question Answering System')
st.write('Upload a PDF file and then ask questions about its content.')

llm, embedding, astra_vector_store, astra_vector_index = initialize_system()

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    raw_text = process_pdf(uploaded_file)
    prepare_text_vector_store(astra_vector_store, raw_text)
    st.write('PDF processed. You can now ask questions.')

question = st.text_input('Ask a question:')
if question:
    answer = query_pdf(astra_vector_index, llm, question)
    st.write(f"Answer: {answer}")


