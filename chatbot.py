import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.chains.question_answering import load_qa_chain

load_dotenv() 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("Please set OPENAI_API_KEY environment variable")
    st.stop()

st.header("Welcome to the Chatbot Interface")

with st.sidebar:
    st.title("Your documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type=["txt", "pdf", "docx"])

# Extract the text
if file is not None:
    if file.type == "application/pdf":
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file.type == "text/plain":
        text = str(file.read(), "utf-8")
    else:
        st.error("Unsupported file type. Please upload a PDF or TXT file.")
        text = ""
    # st.write(text)

    # Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)
    # st.write(chunks)

    # Use session state to cache vector store and avoid recreating on every rerun
    file_id = f"{file.name}_{file.size}"
    
    if 'vector_store' not in st.session_state or st.session_state.get('processed_file_id') != file_id:
        try:
            with st.spinner("Creating embeddings and vector store... This may take a moment."):
                # generate embeddings
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                # creating the vector store - FAISS
                vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                st.session_state.vector_store = vector_store
                st.session_state.processed_file_id = file_id
                st.success("✅ Document processed successfully!")
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower() or "insufficient_quota" in error_msg.lower():
                st.error("⚠️ **OpenAI API Quota Error**: You've exceeded your API quota.")
                st.info("💡 **Troubleshooting Steps**:\n"
                       "1. Check your OpenAI billing: https://platform.openai.com/account/billing\n"
                       "2. Verify your payment method is active and valid\n"
                       "3. Wait 5-10 minutes for new credits to be processed\n"
                       "4. Check if you have usage/spending limits set (remove them if needed)\n"
                       "5. Verify the API key has proper permissions")
            else:
                st.error(f"❌ **Error creating embeddings**: {error_msg}")
            st.stop()
    
    vector_store = st.session_state.vector_store

    user_question = st.text_input("Type your question here")

    # do similarity search
    if user_question:
        try:
            # get the relevant documents
            match = vector_store.similarity_search(user_question)
            st.write(match)
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                st.error("⚠️ **Quota Error during search**. Please check your OpenAI account billing.")
            else:
                st.error(f"❌ **Search Error**: {error_msg}")

        # define the LLM
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=1000,
            model="gpt-3.5-turbo"
        )
        
        # Create QA chain
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_question)
        st.write(response)





