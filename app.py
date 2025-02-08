import streamlit as st
import logging
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import ollama
import re

# Configure logging
logging.basicConfig(
    filename="chatbot_logs.log", 
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

model_id = "deepseek-r1:8b"

def process_pdf(uploaded_file):
    if uploaded_file is None:
        return None, None, None

    try:
        file_path = "temp_uploaded.pdf"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        logging.info("PDF uploaded and saved successfully.")

        loader = PyMuPDFLoader(file_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100
        )
        chunks = text_splitter.split_documents(data)
        logging.info(f"PDF split into {len(chunks)} chunks.")

        embeddings = OllamaEmbeddings(model=model_id)
        vectorstore = Chroma.from_documents(
            documents=chunks, embedding=embeddings, persist_directory="./chroma_db"
        )
        retriever = vectorstore.as_retriever()
        logging.info("Vectorstore created and retriever initialized.")

        return text_splitter, vectorstore, retriever
    except Exception as e:
        logging.error(f"Error in processing PDF: {str(e)}")
        return None, None, None


def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ollama_llm_streaming(question, context):
    system_prompt = (
        "You are an AI assistant specialized in answering questions concisely. "
        "Provide clear and direct responses, keeping answers as brief as possible while maintaining accuracy. "
        "If necessary, limit your response to 2-3 sentences."
    )

    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    logging.info("Sending request to Ollama LLM...")

    try:
        response_stream = ollama.chat(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},  # System message to control behavior
                {"role": "user", "content": formatted_prompt}
            ],
            stream=True  # Enable streaming
        )

        final_answer = ""
        for chunk in response_stream:
            content = chunk["message"]["content"]
            final_answer += content
            yield content  # Stream token to UI

        final_answer = re.sub(r"<think>.*?</think>", "", final_answer, flags=re.DOTALL).strip()
        logging.info("Ollama response received successfully.")
    except Exception as e:
        logging.error(f"Error in Ollama LLM response: {str(e)}")
        yield "Error in generating response."

def rag_chain_streaming(question, retriever):
    logging.info(f"Retrieving context for question: {question}")
    retrieved_docs = retriever.invoke(question)
    formatted_content = combine_docs(retrieved_docs)
    logging.info(f"Retrieved {len(retrieved_docs)} relevant documents.")
    return ollama_llm_streaming(question, formatted_content)


# Streamlit UI
st.title("Ask questions about your PDF 📄🤖")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    text_splitter, vectorstore, retriever = process_pdf(uploaded_file)
    if text_splitter:
        st.session_state["retriever"] = retriever
        st.success("✅ PDF processed successfully!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask a question..."):
    if "retriever" in st.session_state:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            response_stream = rag_chain_streaming(question, st.session_state["retriever"])
            response_text = st.write_stream(response_stream)

        st.session_state.messages.append({"role": "assistant", "content": response_text})
    else:
        st.warning("⚠️ Please upload and process a PDF first.")
