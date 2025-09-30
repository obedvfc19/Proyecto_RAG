# crear_vectordb_general.py (versión multiformato)

import os
import sys
import logging
from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# --- CONFIGURACIÓN DE LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler("db_creation_general.log"), logging.StreamHandler(sys.stdout)])

# --- 1. Definir la Carpeta de Documentos ---
DOCUMENTS_PATH = "./documentos_fuente"

if not os.path.exists(DOCUMENTS_PATH):
    logging.error(f"La carpeta de documentos no existe: {DOCUMENTS_PATH}")
    sys.exit()

# --- 2. Cargar Documentos (LÓGICA ACTUALIZADA PARA PDF Y TXT) ---
logging.info("Iniciando la carga de documentos multiformato...")
documentos_cargados = []
for filename in os.listdir(DOCUMENTS_PATH):
    filepath = os.path.join(DOCUMENTS_PATH, filename)
    
    if filename.endswith(".pdf"):
        try:
            reader = PdfReader(filepath)
            logging.info(f"Leyendo archivo PDF: {filename}...")
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    metadata = {"source": filename, "page": page_num + 1}
                    doc = Document(page_content=text, metadata=metadata)
                    documentos_cargados.append(doc)
        except Exception as e:
            logging.error(f"Error procesando el PDF {filename}: {e}")
            
    elif filename.endswith(".txt"):
        try:
            logging.info(f"Leyendo archivo de Texto: {filename}...")
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            if text:
                # Los archivos de texto no tienen páginas, solo la fuente.
                metadata = {"source": filename}
                doc = Document(page_content=text, metadata=metadata)
                documentos_cargados.append(doc)
        except Exception as e:
            logging.error(f"Error procesando el archivo de texto {filename}: {e}")

if not documentos_cargados:
    logging.error("No se pudieron cargar documentos. Verifique la carpeta de origen.")
    sys.exit()

logging.info(f"Se cargó el contenido de {len(os.listdir(DOCUMENTS_PATH))} archivos.")

# --- 3. Chunking Genérico (sin cambios) ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks_finales = text_splitter.split_documents(documentos_cargados)
logging.info(f"Se crearon {len(chunks_finales)} chunks en total.")

# --- 4. Creación de la Base de Datos Vectorial (sin cambios) ---
logging.info("Creando la base de datos vectorial final...")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_store = Chroma.from_documents(
    documents=chunks_finales, 
    embedding=embeddings,
    persist_directory="./chroma_db_final"
)

logging.info("\n¡Éxito! La base de datos vectorial ha sido creada/actualizada con los nuevos documentos.")
