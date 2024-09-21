import streamlit as st #Streamlit for UI Interface
from PyPDF2 import PdfReader #To read PDF Files

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
        st.write(text)

#Break it into chunks


