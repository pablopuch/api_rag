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

3. Configurar la API Key de LangChain

    El proyecto utiliza la API de LangChain para descargar el prompt. Necesitarás configurar tu API key en el entorno virtual. En tu sistema operativo, puedes establecer variables de entorno de manera manual.

    Configura la API Key como una variable de entorno:

    En Windows:

    ```bash
    setx API_KEY "tu_api_key_aqui"
    ```
    En macOS/Linux:

    ```bash
    export API_KEY="tu_api_key_aqui"
    ```


4. Instalar las dependencias:

   ```
    pip install -r requirements.txt
   ```
   
5. Ejecutar la API:

   ```
   fastapi dev main.py
   ```
# Ollama

<div align="center">
 <img alt="ollama" height="200px" src="https://github.com/ollama/ollama/assets/3325447/0d0b44e2-8f4a-4e99-9b52-a5c1c741c8f7">
</div>

### macOS

[Download](https://ollama.com/download/Ollama-darwin.zip)

### Windows preview

[Download](https://ollama.com/download/OllamaSetup.exe)


## Quickstart

To run and chat with [Llama 3.1](https://ollama.com/library/llama3.1):

```
ollama run llama3.1
```

## Model library

Ollama supports a list of models available on [ollama.com/library](https://ollama.com/library 'ollama model library')

Here are some example models that can be downloaded:

| Model              | Parameters | Size  | Download                       |
| ------------------ | ---------- | ----- | ------------------------------ |
| Llama 3.1          | 8B         | 4.7GB | `ollama run llama3.1`          |
| Llama 3.1          | 70B        | 40GB  | `ollama run llama3.1:70b`      |
| Llama 3.1          | 405B       | 231GB | `ollama run llama3.1:405b`     |
| Phi 3 Mini         | 3.8B       | 2.3GB | `ollama run phi3`              |
| Phi 3 Medium       | 14B        | 7.9GB | `ollama run phi3:medium`       |
| Gemma 2            | 2B         | 1.6GB | `ollama run gemma2:2b`         |
| Gemma 2            | 9B         | 5.5GB | `ollama run gemma2`            |
| Gemma 2            | 27B        | 16GB  | `ollama run gemma2:27b`        |
| Mistral            | 7B         | 4.1GB | `ollama run mistral`           |
| Moondream 2        | 1.4B       | 829MB | `ollama run moondream`         |
| Neural Chat        | 7B         | 4.1GB | `ollama run neural-chat`       |
| Starling           | 7B         | 4.1GB | `ollama run starling-lm`       |
| Code Llama         | 7B         | 3.8GB | `ollama run codellama`         |
| Llama 2 Uncensored | 7B         | 3.8GB | `ollama run llama2-uncensored` |
| LLaVA              | 7B         | 4.5GB | `ollama run llava`             |
| Solar              | 10.7B      | 6.1GB | `ollama run solar`             |

> [!NOTE]
> You should have at least 8 GB of RAM available to run the 7B models, 16 GB to run the 13B models, and 32 GB to run the 33B models.

## Customize a model

### Import from GGUF

Ollama supports importing GGUF models in the Modelfile:

1. Create a file named `Modelfile`, with a `FROM` instruction with the local filepath to the model you want to import.

   ```
   FROM ./vicuna-33b.Q4_0.gguf
   ```

2. Create the model in Ollama

   ```
   ollama create example -f Modelfile
   ```

3. Run the model

   ```
   ollama run example
   ```

### Import from PyTorch or Safetensors

See the [guide](docs/import.md) on importing models for more information.

### Customize a prompt

Models from the Ollama library can be customized with a prompt. For example, to customize the `llama3.1` model:

```
ollama pull llama3.1
```

Create a `Modelfile`:

```
FROM llama3.1

# set the temperature to 1 [higher is more creative, lower is more coherent]
PARAMETER temperature 1

# set the system message
SYSTEM """
You are Mario from Super Mario Bros. Answer as Mario, the assistant, only.
"""
```

Next, create and run the model:

```
ollama create mario -f ./Modelfile
ollama run mario
>>> hi
Hello! It's your friend Mario.
```

For more examples, see the [examples](examples) directory. For more information on working with a Modelfile, see the [Modelfile](docs/modelfile.md) documentation.

## CLI Reference

### Create a model

`ollama create` is used to create a model from a Modelfile.

```
ollama create mymodel -f ./Modelfile
```

### Pull a model

```
ollama pull llama3.1
```

> This command can also be used to update a local model. Only the diff will be pulled.

### Remove a model

```
ollama rm llama3.1
```

### Copy a model

```
ollama cp llama3.1 my-model
```

### Multiline input

For multiline input, you can wrap text with `"""`:

```
>>> """Hello,
... world!
... """
I'm a basic program that prints the famous "Hello, world!" message to the console.
```

### Multimodal models

```
ollama run llava "What's in this image? /Users/jmorgan/Desktop/smile.png"
The image features a yellow smiley face, which is likely the central focus of the picture.
```

### Pass the prompt as an argument

```
$ ollama run llama3.1 "Summarize this file: $(cat README.md)"
 Ollama is a lightweight, extensible framework for building and running language models on the local machine. It provides a simple API for creating, running, and managing models, as well as a library of pre-built models that can be easily used in a variety of applications.
```

### Show model information

```
ollama show llama3.1
```

### List models on your computer

```
ollama list
```

### Start Ollama

`ollama serve` is used when you want to start ollama without running the desktop application.

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
