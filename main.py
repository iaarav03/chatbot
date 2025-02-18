import os
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pymongo import MongoClient

# LangChain and related imports
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnablePassthrough

# ======================
# 1. Load Env Variables
# ======================
load_dotenv()
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["mydatabase"]
sessions_collection = db["sessions"]

# ======================
# 2. Create FastAPI App
# ======================
app = FastAPI()

# ======================
# 3. Initialize Embeddings & LLM
# ======================
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-l6-V2")
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.0,
    base_url="http://localhost:11434",
)

# ======================
# 4. System Prompts & Chain Setup
# ======================
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

# ======================
# 5. Session Data in Memory (for Chat History)
# ======================
session_data = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Get or create the in-memory ChatMessageHistory for a session."""
    if session_id not in session_data:
        session_data[session_id] = {}
        session_data[session_id]["chat_history"] = ChatMessageHistory()
    return session_data[session_id]["chat_history"]

# ======================
# 6. /upload Endpoint
# ======================
@app.post("/upload")
async def upload_files(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    # Load and split docs
    documents = []
    for uploaded_file in files:
        temp_path = f"./temp_{uploaded_file.filename}"
        with open(temp_path, "wb") as f:
            content = await uploaded_file.read()
            f.write(content)
        try:
            if uploaded_file.filename.lower().endswith(".pdf"):
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

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)

    # Store splits in Mongo under this session
    from langchain.docstore.document import Document
    splits_dict = []
    for doc in splits:
        splits_dict.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata
        })

    sessions_collection.update_one(
        {"session_id": session_id},
        {"$set": {"documents": splits_dict}},
        upsert=True
    )

    return {"message": "Files processed and documents stored for session.", "session_id": session_id}

# ======================
# 7. /ask Endpoint
# ======================
@app.post("/ask")
async def ask_question(session_id: str, question: str):
    # Retrieve docs from Mongo
    session_record = sessions_collection.find_one({"session_id": session_id})
    if not session_record or "documents" not in session_record:
        raise HTTPException(status_code=400, detail="Session not found or no documents uploaded.")

    # Rebuild Document objects
    from langchain.docstore.document import Document
    docs = []
    for d in session_record["documents"]:
        docs.append(Document(page_content=d["page_content"], metadata=d["metadata"]))

    # Create vectorstore and retriever
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Build a history-aware retriever
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Create chain
    chain = create_chain(history_aware_retriever)

    # Store chain in memory if you like
    if session_id not in session_data:
        session_data[session_id] = {}
    session_data[session_id]["chain"] = chain

    # Use RunnableWithMessageHistory for conversation
    chat_history = get_session_history(session_id)
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

# ======================
# 8. Run the App
# ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
