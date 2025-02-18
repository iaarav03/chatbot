import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from dotenv import load_dotenv

# Updated LangChain imports
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama
# Use the HuggingFaceEmbeddings from the langchain-huggingface package
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

app = FastAPI()

# Initialize embeddings using the updated HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-l6-V2")

# Initialize LLM (using ChatOllama)
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.0,
    base_url="http://localhost:11434",
)

# Global session store for vectorstores, chains, and chat histories
session_data = {}

def get_session_history(session: str) -> ChatMessageHistory:
    if session not in session_data:
        session_data[session] = {}
        session_data[session]["chat_history"] = ChatMessageHistory()
    return session_data[session]["chat_history"]

# Prompt to reformulate questions based on chat history
contextualize_q_system_prompt = """
Given a chat history and latest user question,
which might reference context in chat history,
reformulate it into a standalone question that can be understood
without that chat history. Do not answer the question, just reformulate it
if needed and otherwise return it as is.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# System prompt for the QA chain (medical information assistant)
system_prompt = """
You are a specialized medical information assistant designed to provide accurate, document-based medical information. Your responses must adhere to these strict guidelines:

1. SOURCE ATTRIBUTION:
   - Every piece of information must be explicitly linked to the source document
   - Use format: [Document Name, Page/Section] for each reference
   - If information is not found in the provided documents, state: "I cannot find information about this in the provided documents."

2. RESPONSE STRUCTURE:
   - Start with: "Based on the provided documents:"
   - Organize information under clear headings
   - Use bullet points for listing information

3. ACCURACY REQUIREMENTS:
   - Only provide information explicitly stated in the documents
   - Do not make interpretations or draw conclusions not directly stated
   - Do not combine information from different sources to make new inferences
   - If information is incomplete, state what is missing

4. MEDICAL INFORMATION HANDLING:
   - Present medical terms exactly as they appear in the documents
   - Include units of measurement exactly as specified
   - For medications, always include dosage/frequency if mentioned
   - For procedures, maintain exact technical terminology

5. LIMITATIONS:
   - Never provide medical advice or recommendations
   - Do not speculate about treatments or diagnoses
   - If asked about medical advice, respond: "I cannot provide medical advice. Please consult a healthcare professional."

Context: {context}
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

def create_chain(history_aware_retriever):
    retrieval_chain = RunnablePassthrough.assign(
        context=history_aware_retriever.with_config(run_name="retrieve_documents")
    )
    response_chain = retrieval_chain.assign(
        answer=create_stuff_documents_chain(llm, qa_prompt)
    )
    return response_chain.with_config(run_name="retrieval_chain")

@app.post("/upload")
async def upload_files(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    documents = []
    # Process each uploaded file
    for uploaded_file in files:
        temp_path = f"./temp_{uploaded_file.filename}"
        with open(temp_path, "wb") as f:
            content = await uploaded_file.read()
            f.write(content)
        try:
            if uploaded_file.filename.lower().endswith('.pdf'):
                loader = PyPDFLoader(temp_path)
            else:
                loader = TextLoader(temp_path)
            docs = loader.load()
            documents.extend(docs)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    if not documents:
        raise HTTPException(status_code=400, detail="No documents loaded.")
    
    # Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500
    )
    splits = text_splitter.split_documents(documents)
    
    # Create the vectorstore using Chroma
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    # Create retriever and history-aware retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )
    
    chain = create_chain(history_aware_retriever)
    
    # Save vectorstore and chain in the session store
    if session_id not in session_data:
        session_data[session_id] = {}
    session_data[session_id]["vectorstore"] = vectorstore
    session_data[session_id]["chain"] = chain
    
    return {"message": "Files processed and vectorstore created for session.", "session_id": session_id}

@app.post("/ask")
async def ask_question(session_id: str, question: str):
    if session_id not in session_data or "chain" not in session_data[session_id]:
        raise HTTPException(status_code=400, detail="Session not found or files not uploaded.")
    
    chat_history = get_session_history(session_id)
    chain = session_data[session_id]["chain"]
    
    # Create the conversational chain with message history
    conversational_rag_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    response = conversational_rag_chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}}
    )
    
    result = {
        "answer": response.get("answer", ""),
        "sources": [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "content": doc.page_content
            } for doc in response.get("context", [])
        ]
    }
    
    return JSONResponse(content=result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
