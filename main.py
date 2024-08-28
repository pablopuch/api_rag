from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain import hub
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
import os
import shutil
import asyncio

app = FastAPI(
    title="Mi API de Documentos",
    description="API para subir y procesar documentos PDF usando LLMs",
    version="1.0.0",
)

api_key = os.getenv('API_KEY')
if not api_key:
    raise ValueError("API_KEY no está configurada correctamente.")

# Directorio para almacenar archivos PDF cargados
UPLOAD_DIRECTORY = "uploaded_docs"

# Crear directorio si no existe
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Variable global para almacenar la cadena de QA
qa_chain = None

# Cargar el modelo de embeddings una vez, fuera de las funciones
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Cargar el contenido de todos los archivos PDF en una carpeta
async def load_documents_async(folder_path):
    loop = asyncio.get_event_loop()
    all_documents = []

    tasks = [
        loop.run_in_executor(None, PyPDFLoader(os.path.join(folder_path, filename)).load)
        for filename in os.listdir(folder_path) if filename.endswith('.pdf')
    ]
    
    loaded_docs = await asyncio.gather(*tasks)
    
    for docs in loaded_docs:
        all_documents.extend(docs)
    
    return all_documents

# Dividir el texto en fragmentos semánticos
def split_text(data):
    semantic_chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    return semantic_chunker.create_documents([d.page_content for d in data])

# Crear el vectorstore a partir de los fragmentos de texto
def create_vectorstore(splits):
    return Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory='./vectordb2')

# Descargar el prompt de RAG
def load_prompt():
    return hub.pull("llama-rag", api_key=api_key)

# Configurar el modelo de lenguaje Llama3.1 con Ollama
def configure_llm():
    return Ollama(model="llama3.1:latest", verbose=True)

# Configurar la cadena de preguntas y respuestas con recuperación
def create_qa_chain(llm, vectorstore, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )

# Inicializar el sistema: cargar documentos, crear vectorstore y cadena de QA
async def initialize_system():
    global qa_chain
    
    # 1. Cargar documentos desde la carpeta especificada
    data = await load_documents_async(UPLOAD_DIRECTORY)
    
    # 2. Dividir en fragmentos
    all_splits = split_text(data)
    
    # 3. Crear el vectorstore
    vectorstore = create_vectorstore(all_splits)
    
    # 4. Cargar el prompt
    prompt = load_prompt()
    
    # 5. Configurar el LLM
    llm = configure_llm()
    
    # 6. Configurar la cadena de QA
    qa_chain = create_qa_chain(llm, vectorstore, prompt)

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload-docs/", summary="Subir documentos", description="Sube archivos PDF y reinicia el sistema.")
async def upload_docs(files: List[UploadFile], background_tasks: BackgroundTasks):
    # Guardar los archivos PDF en el directorio
    for file in files:
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    
    # Re-inicializar el sistema para procesar los nuevos documentos
    background_tasks.add_task(initialize_system)
    
    return {"message": "Documents uploaded. Processing in background."}

@app.post("/ask-question/", summary="Hacer una pregunta", description="Haz una pregunta sobre los documentos procesados.")
async def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="La pregunta no puede estar vacía.")
    if qa_chain is None:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    # Obtener la respuesta a la pregunta usando el QA chain
    result = qa_chain({"query": request.question})
    
    return {"response": result["result"]}

# Ejecutar el servidor de FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)