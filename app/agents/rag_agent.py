from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os
from fastapi import UploadFile
import tempfile
from pdfminer.high_level import extract_text

class FileAgent:
    """Handles document-based queries using RAG."""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)

    def parse_file(self, file: UploadFile) -> str:
        """Extract text from the uploaded file."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.file.read())
            temp_file_path = temp_file.name
        extracted_text = extract_text(temp_file_path)
        os.remove(temp_file_path)
        return extracted_text

    
    def process_file_query(self, query: str, file: UploadFile) -> str:
        # Step 1: Extract text from the file
        document_text = self.parse_file(file)

        # Step 2: Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(document_text)

        # Step 3: Create embeddings and store in ChromaDB
        vector_store = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            persist_directory="./db/chroma"  # Set the folder for persistent storage
        )

        # Step 4: Save the ChromaDB index
        vector_store.persist()

        # Step 5: Create a RetrievalQA chain
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=False
        )

        # Step 6: Query the chain
        response = qa_chain.run(query)
        return response
    

  