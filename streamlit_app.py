import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

import os
import PyPDF2
from langchain.docstore.document import Document

def read_pdf_files(directory):
    docs = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdf'):
                pdf_file_path = os.path.join(root, file)
                with open(pdf_file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfFileReader(f)
                    text = ''
                    for page in range(pdf_reader.numPages):
                        text += pdf_reader.getPage(page).extractText()
                    doc = Document(text, metadata={"filename": file})
                    docs.append(doc)
    return docs

docs = read_pdf_files('./data')

st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets["OPENAI_API_KEY"]
st.title("Chat with the Streamlit docs, powered by LlamaIndex 💬🦙")
st.info("Check out the full tutorial to build this app in our [blog post](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)", icon="📃")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about Streamlit's open-source Python library!",
        }
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    docs = read_pdf_files('./data')
    Settings.llm = OpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        system_prompt="""You are an expert on 
        the Streamlit Python library and your 
        job is to answer technical questions. 
        Assume that all questions are related 
        to the Streamlit Python library. Keep 
        your answers technical and based on 
        facts – do not hallucinate features.""",
    )
    index = VectorStoreIndex.from_documents(docs)
    return index


index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )

if prompt := st.chat_input(
    "Ask a question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        # Add response to message history
        st.session_state.messages.append(message)