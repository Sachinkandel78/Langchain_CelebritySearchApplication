import os
import streamlit as st
from constants import groq_api_key
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain

from langchain_classic.chains import SequentialChain
from langchain_classic.memory import ConversationBufferMemory

os.environ["GROQ_API_KEY"] = groq_api_key

## streamlit framework
st.title("Celebrity Search Application with LangChain and Groq API")
input_text = st.text_input("Search for a celebrity")

## Prompt template for the LLM

first_input_prompt = PromptTemplate(
    input_variables=["name"],
    template="Tell me about celebrity {name}.",
)

## memory for the LLM
person_memory = ConversationBufferMemory(input_key="name", memory_key="person", output_key="person")
dob_memory = ConversationBufferMemory(input_key="person", memory_key="dob", output_key="dob")
description_memory = ConversationBufferMemory(input_key="dob", memory_key="description", output_key="description")

## groq LLM
llm = ChatGroq( model="llama-3.1-8b-instant",  temperature=0.8 )
chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key="person", memory=person_memory)

## Prompt template for the LLM

second_input_prompt = PromptTemplate(
    input_variables=["person"],
    template="when was {person} born?",
)
chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key="dob", memory=dob_memory)
third_input_prompt = PromptTemplate(
    input_variables=["dob"],
    template="Mention 5 major events happened around {dob} in the world.",
)
chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose=True, output_key="description", memory=description_memory)

parent_chain = SequentialChain(
    chains=[chain, chain2, chain3], input_variables=['name'], output_variables=['person','dob','description'], verbose=True)

if input_text:
    st.write(parent_chain.invoke({"name": input_text}))

    with st.expander('Person name'):
        st.write(person_memory.buffer)
    

    with st.expander('Description'):
        st.write(description_memory.buffer)

// Extentension of langchain to rag 
