## Integrate our code Groq API with LangChain and Streamlit

import os
import streamlit as st
from constants import groq_api_key
from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"] = groq_api_key


##Streamlit framework
st.title("Celebrity Search Application with LangChain and Groq")
input_text = st.text_input("Search for a celebrity")

## Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.8
)

if input_text:
    response = llm.invoke(input_text)
    st.write(response.content)