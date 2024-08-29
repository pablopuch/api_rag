from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from langchain.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from pydantic import BaseModel
from typing import List
from langchain import hub
import asyncio
import os


app = FastAPI(
    title="Mi API de Documentos",
    description="API para subir y procesar documentos PDF usando LLMs",
    version="1.0.0",
)

api_key = os.getenv('API_KEY')
if not api_key:
    raise ValueError("API_KEY no está configurada correctamente.")

UPLOAD_DIRECTORY = "uploaded_docs"

import pathlib
pathlib.Path(UPLOAD_DIRECTORY).mkdir(parents=True, exist_ok=True)

qa_chain = None
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

async def load_documents_async(folder_path):
    loop = asyncio.get_event_loop()
    all_documents = []

    tasks = [
        loop.run_in_executor(None, PyPDFLoader(str(file_path)).load)
        for file_path in pathlib.Path(folder_path).glob('*.pdf')
    ]
    
    loaded_docs = await asyncio.gather(*tasks)
    
    for docs in loaded_docs:
        all_documents.extend(docs)
    
    return all_documents

def split_text(data):
    semantic_chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    return semantic_chunker.create_documents([d.page_content for d in data])

def create_vectorstore(splits):
    return Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory='./vectordb2')

# Descargar el prompt de RAG
def load_prompt():
    return hub.pull("llama-rag", api_key=api_key)

def configure_llm():
    return Ollama(model="llama3.1:latest", verbose=True)


def create_qa_chain(llm, vectorstore, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )

async def initialize_system():
    global qa_chain
    
    data = await load_documents_async(UPLOAD_DIRECTORY)
    all_splits = split_text(data)
    vectorstore = create_vectorstore(all_splits)
    prompt = load_prompt()
    llm = configure_llm()
    qa_chain = create_qa_chain(llm, vectorstore, prompt)

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload-docs/", summary="Subir documentos", description="Sube archivos PDF y reinicia el sistema.")
async def upload_docs(files: List[UploadFile], background_tasks: BackgroundTasks):
    for file in files:
        file_path = pathlib.Path(UPLOAD_DIRECTORY) / file.filename
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
    
    background_tasks.add_task(initialize_system)
    
    return {"message": "Documents uploaded. Processing in background."}

@app.post("/ask-question/", summary="Hacer una pregunta", description="Haz una pregunta sobre los documentos procesados.")
async def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="La pregunta no puede estar vacía.")
    
    # Inicializar el sistema antes de hacer la pregunta
    try:
        await initialize_system()
    except Exception as e:
        print(f"Error inicializando el sistema: {e}")  # Para depuración
        raise HTTPException(status_code=500, detail=f"Error inicializando el sistema: {str(e)}")
    
    if qa_chain is None:
        raise HTTPException(status_code=500, detail="Error al inicializar el sistema.")
    
    try:
        result = qa_chain.invoke({"query": request.question})
        return {"response": result.get("result", "No result returned")}
    except Exception as e:
        print(f"Error procesando la pregunta: {e}")  # Para depuración
        raise HTTPException(status_code=500, detail=f"Error procesando la pregunta: {str(e)}")
