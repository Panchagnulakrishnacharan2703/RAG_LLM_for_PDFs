import streamlit as st
import os
import tempfile
from logs.logger import logger
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

@st.cache_resource()
def load_pdf(file_path):
    logger.info("Loading PDF from path: %s", file_path)
    loader = UnstructuredPDFLoader(file_path=file_path)
    data = loader.load()
    logger.info("PDF loaded successfully")
    return data

@st.cache_resource()
def create_vector_db(_data):  
    logger.info("Creating vector database from data")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
    chunks = text_splitter.split_documents(_data)  
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
        collection_name="local-rag"
    )
    logger.info("Vector database created successfully with %d chunks", len(chunks))
    return vector_db

@st.cache_resource()
def create_llm(model_name):
    logger.info("Creating LLM with model name: %s", model_name)
    llm = ChatOllama(model=model_name)
    logger.info("LLM created successfully")
    return llm

@st.cache_resource()
def create_retriever(_vector_db, _llm):
    logger.info("Creating retriever")
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )
    retriever = MultiQueryRetriever.from_llm(
        _vector_db.as_retriever(),
        _llm,
        prompt=QUERY_PROMPT
    )
    logger.info("Retriever created successfully")
    return retriever

@st.cache_resource()
def create_chain(_retriever, _llm):
    logger.info("Creating chain")
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": _retriever, "question": RunnablePassthrough()}
        | prompt
        | _llm
        | StrOutputParser()
    )
    logger.info("Chain created successfully")
    return chain

def main():
    st.title("RAG Model for PDF Question Answering")
    st.header("Upload a PDF file")
    file_path = st.file_uploader("Select a PDF file", type=["pdf"])

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if file_path:
        logger.info("PDF file uploaded: %s", file_path.name)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_file_path = os.path.join(tmp_dir, file_path.name)
            with open(tmp_file_path, 'wb') as f:
                f.write(file_path.getbuffer())
            logger.info("PDF file saved to temporary directory: %s", tmp_file_path)

            data = load_pdf(tmp_file_path)
            st.write("PDF loaded successfully!")
            
            vector_db = create_vector_db(data)
            st.write("Vector database created successfully!")
            
            llm = create_llm("mistral")
            st.write("LLM model loaded successfully!")
            
            retriever = create_retriever(vector_db, llm)
            st.write("Retriever created successfully!")
            
            chain = create_chain(retriever, llm)
            st.write("Chain created successfully!")

            st.header("Ask a question about the PDF")
            question = st.text_input("Enter a question")

            if question:
                logger.info("User question: %s", question)
                answer = chain.invoke(question)
                st.session_state.chat_history.append((question, answer))
                st.write("Answer:", answer)
                logger.info("Answer: %s", answer)

            st.header("Chat History")
            for q, a in st.session_state.chat_history:
                st.write(f"**Question:** {q}")
                st.write(f"**Answer:** {a}")

    else:
        st.write("Please upload a PDF file")
        logger.warning("No PDF file uploaded")

if __name__ == "__main__":
    main()