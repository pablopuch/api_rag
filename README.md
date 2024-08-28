# API de Procesamiento de Documentos con LLMs

### Descripción General

Esta API permite subir y procesar documentos PDF para posteriormente realizar consultas sobre su contenido utilizando modelos de lenguaje natural (LLMs). El sistema está diseñado para ser flexible y escalable, permitiendo cambiar los modelos de embeddings y el modelo de lenguaje utilizado para las consultas, además de poder ampliar la funcionalidad para soportar otros tipos de documentos como Word, Excel, etc.

## Instalación

### Requisitos Previos

Asegúrate de tener Python 3.8+ instalado en tu sistema. También necesitarás un entorno virtual para manejar las dependencias del proyecto.

### Pasos de Instalación

1. Clonar el repositorio de la API:

   ```
   git clone https://github.com/pablopuch/api_rag.git
    ```


2. Crear un entorno virtual:

   ```
    py -m venv env
    source env/bin/activate   # En Windows usa: .\env\Scripts\activate
   ```

3. Instalar las dependencias:

   ```
    pip install -r requirements.txt
   ```

4. Configurar la variable de entorno API_KEY:

    Es necesario definir la variable de entorno API_KEY con tu clave personal para acceder al modelo de LLM en Ollama:

   ```
   pip install -r requirements.txt
   ```

5. Ejecutar la API:

   ```
   fastapi dev main.py
   ```


# Uso de la API

### Endpoints Disponibles

```http
  POST /upload-docs/
```

| Type     | Description                |
| :------- | :------------------------- |
| `file` | Permite subir archivos PDF al servidor para ser procesados. Los archivos se guardan en el directorio especificado y el sistema se re-inicializa en segundo plano para cargar y procesar los nuevos documentos. |


```http
  POST /ask-question/
```

| Type     | Description                       |
| :------- | :-------------------------------- |
| `string` | Permite hacer preguntas sobre el contenido de los documentos procesados. La pregunta se resuelve utilizando un modelo de lenguaje natural preconfigurado. |

# Configuración Avanzada

### Cambiar el Modelo de Ollama

Para cambiar el modelo de lenguaje utilizado (por ejemplo, de llama3.1 a otro modelo disponible), puedes modificar la función configure_llm en el archivo principal (main.py):

```python
def configure_llm():
    return Ollama(model="NOMBRE_DEL_MODELO:latest", verbose=True)
```

Asegúrate de que el modelo esté disponible en la plataforma de Ollama y que tu API Key tenga acceso a dicho modelo.

### Cambiar el Modelo de Embeddings

Si prefieres usar un modelo diferente para los embeddings (por ejemplo, uno más grande o específico para otro idioma), puedes cambiar la inicialización de HuggingFaceEmbeddings en la configuración global:

```python
embeddings = HuggingFaceEmbeddings(model_name='NUEVO_MODELO_DE_EMBEDDINGS')
```

Modelos alternativos podrían incluir sentence-transformers/all-mpnet-base-v2 o cualquier otro disponible en Hugging Face.

### Soporte para Otros Tipos de Archivos

Para añadir soporte a otros tipos de archivos como Word o Excel, necesitarás agregar nuevos loaders. Aquí te mostramos un ejemplo simple de cómo podrías extender la funcionalidad para soportar documentos de Word:

1. Instala la librería necesaria para manejar archivos Word:

```
pip install python-docx
```

2. Modifica la función load_documents_async para incluir soporte a archivos .docx:

```python
from langchain.document_loaders import PyPDFLoader, DocxLoader

async def load_documents_async(folder_path):
    loop = asyncio.get_event_loop()
    all_documents = []

    tasks = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            tasks.append(loop.run_in_executor(None, PyPDFLoader(os.path.join(folder_path, filename)).load))
        elif filename.endswith('.docx'):
            tasks.append(loop.run_in_executor(None, DocxLoader(os.path.join(folder_path, filename)).load))
    
    loaded_docs = await asyncio.gather(*tasks)
    
    for docs in loaded_docs:
        all_documents.extend(docs)
    
    return all_documents
```

Con esta configuración, la API también procesará documentos de Word (.docx). Puedes seguir el mismo enfoque para agregar soporte a otros formatos como Excel, utilizando loaders específicos para cada tipo de archivo.
