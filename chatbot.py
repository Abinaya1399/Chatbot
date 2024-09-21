import streamlit as st #Streamlit for UI Interface
from PyPDF2 import PdfReader #Reads PDF Files and Extracts Text
from langchain.text_splitter import RecursiveCharacterTextSplitter #Breaks text into chunks
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
#pip install tiktoken
#pip install openai

#Upload PDF files
st.header("My First Chatbot")
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

#Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text=""
    for page in pdf_reader.pages:
        text+=page.extract_text()
        # st.write(text)

    #Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators = "\n",
        chunk_size = 1000,
        chunk_overlap = 150,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    # st.write(chunks)

    #Generating embeddings - Use OpenAI services & the OpenAI secret key generated
    #OPENAI_API_KEY - generate secret key from 'https://platform.openai.com/api-keys'
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    #Creating Vector Store - FAISS (Facebook AI Semantic Search)
    vector_store = FAISS.from_texts(chunks, embeddings)

    #Get User Question
    user_question = st.text_input("Type your question here")

    #Do Similarity Search
    #1. User question(text format) -> embeddings
    #2. Find similarity between user question's embeddings and vector store's embeddings
    #3. Corresponding highly ranked chunks will be returned as match
    if user_question:
        match = vector_store.similarity_search(user_question)
        st.write(match)

    #Output Results
