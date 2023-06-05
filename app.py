# Run this app with command "streamlit run app.py"
import os 
# Importing OpenAI as main LLM service
from langchain.llms import OpenAI
# Bring in streamlit for UI/app interface
import streamlit as st
# Bringing in PDF document loaders...there's other ones as well.
from langchain.document_loaders import PyPDFLoader
# Importing chroma as the vector store
from langchain.vectorstores import Chroma



# This can be replaced with other LLM providers
os.environ["OPEN_AI_KEY"] = ""

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, openai_api_key=os.environ["OPEN_AI_KEY"], verbose=True) 

# Set path to the document
avgo_document = os.path.relpath('Docs/broadcom_annual_report.pdf')

# Create a loader for PDF's
loader = PyPDFLoader(avgo_document)

# Split pages from pdf
pages = loader.load_and_split()

# Load documents into vector database aka ChromaDB
store = Chroma.from_documents(pages, collection_name='broadcom_annual_report')

# Create a text input box for the user.
prompt = st.text_input("Input your prompt here")

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    response = llm(prompt)
    # Write it out to the screen
    st.write(response)

    # Use streamlit expander
    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt)
        st.write(search[0][0].page_content)