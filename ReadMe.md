# RAG Model for PDF Question Answering
This repository contains a Streamlit application that leverages Retrieval-Augmented Generation (RAG) to answer questions based on the content of uploaded PDF files. The application processes the PDF, creates embeddings, stores them in a vector database, and uses a language model to generate responses to user queries.

## Directory Structure
`````

RAG-PDF-QA/
├── logs/
│   └── logger.py                       # Handles logging for the application.
├── main.py                             # The main entry point for the Streamlit application.
├── requirements.txt                    # Text file listing project dependencies.
├── README.md                           # Documentation for the project.

`````
## Architecture
![](https://github.com/Panchagnulakrishnacharan2703/RAG_LLM_for_PDFs/blob/main/pics/LLM_ARCHITECTURE.jpg)

## Features
* PDF file upload and processing
* Creation of vector database from PDF content
* Use of language model to generate answers to user questions
* Maintenance of chat history for user questions and answers

## Technologies used
* Streamlit
* LangChain
* Ollama Embeddings
* Chroma Vector Store

## How to use
* Clone the repository: 
```bash
    git clone https://github.com/yourusername/RAG-PDF-QA.git
    cd RAG-PDF-QA
```

* Install the required packages:
```bash    
    pip install -r requirements.txt
```

* Run the Streamlit app:
```bash    
    streamlit run main.py
```

* Open the Streamlit app in your browser.

* Upload a PDF file using the file uploader.

* Enter a question about the content of the PDF.

* View the generated answer and the chat history.
