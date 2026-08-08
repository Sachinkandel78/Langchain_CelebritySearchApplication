import os
import streamlit as st
from constants import groq_api_key
from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"] = groq_api_key

st.title("LangChain Demo with Groq API")

input_text = st.text_input("Search the topic you want")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.8
)

if input_text:
    response = llm.invoke(input_text)
    st.write(response.content)