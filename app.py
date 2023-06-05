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
# Import API Key
from api_file import llm_api_key
# Import more vector store stuff from langchain
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

os.environ["OPENAI_API_KEY"] = llm_api_key

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True) 

# Set path to the document
avgo_document = os.path.relpath('Docs/broadcom_annual_report_2022.pdf')

# Create a loader for PDF's
loader = PyPDFLoader(avgo_document)

# Split pages from pdf
pages = loader.load_and_split()

# Load documents into vector database aka ChromaDB
store = Chroma.from_documents(pages, collection_name='broadcom_annual_report')

# Create vectorstore info object: :like a metadata repo ?
vectorstore_info = VectorStoreInfo(
    name="broadcom_annual_report",
    description="2022 Broadcom annual report as a pdf",
    vectorstore=store
)

# Converting the document store into a langchain toolkit
# This makes the PDF available as a tool to langchain
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end langchain
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)



# Create a text input box for the user.
prompt = st.text_input("Input your prompt here")

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    # response = llm(prompt)

    # Using the document agent instead of the raw llm
    response = agent_executor.run(prompt)

    # ...and write it out to the screen
    st.write(response)

    # Use streamlit expander
    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt)
        st.write(search[0][0].page_content)