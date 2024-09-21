#Streamlit for UI Interface
import streamlit as st

#Upload PDF files
st.header("My First Chatbot")
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")
