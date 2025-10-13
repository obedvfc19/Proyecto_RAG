import gradio as gr
import logging
import sys
import os
import re
import subprocess

# Importaciones para Búsqueda Híbrida
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.docstore.document import Document

# Importaciones estándar del proyecto
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# --- TEMA PERSONALIZADO DE GRADIO ---
theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="slate",
).set(
    body_background_fill="#0E1117",
    body_text_color="#FFFFFF",
    button_primary_background_fill="#1E88E5",
    button_primary_text_color="#FFFFFF",
    block_background_fill="#1C212B",
    block_border_width="0px",
    block_title_text_color="#FFFFFF",
    input_background_fill="#262D3B",
    input_border_width="0px",
    input_placeholder_color="#9CA3AF"
)

# --- CSS PERSONALIZADO ---
css_styles = """
/* Estilo general del chatbot */
#chatbot { box-shadow: none !important; border: none !important; }
/* Burbujas del chat */
.message-bubble { border-radius: 15px !important; box-shadow: 0px 2px 4px rgba(0,0,0,0.1) !important; }
/* Burbuja del usuario */
.user-message > .message-bubble { background-color: #1E88E5 !important; color: white !important; }
/* Burbuja del bot */
.bot-message > .message-bubble { background-color: #262D3B !important; color: #E5E7EB !important; }
"""

# --- CONFIGURACIÓN DE LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler("notaria_debug.log"), logging.StreamHandler(sys.stdout)])

# --- CONFIGURACIÓN INICIAL DE COMPONENTES ---
logging.info("Iniciando NotarIA v5 (Búsqueda Híbrida)...")

# Se mantiene el modelo Llama 3 8B actual
llm = OllamaLLM(model="llama3") 
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

logging.info("Cargando base de datos vectorial...")
vector_db = Chroma(persist_directory="./chroma_db_final", embedding_function=embeddings)

logging.info("Configurando el sistema de búsqueda híbrida...")
docs = vector_db.get()
all_docs = [Document(page_content=content, metadata=metadata or {}) for content, metadata in zip(docs['documents'], docs['metadatas'])]
logging.info(f"Se cargaron {len(all_docs)} documentos en memoria para el retriever BM25.")
bm25_retriever = BM25Retriever.from_documents(all_docs)
bm25_retriever.k = 2
semantic_retriever = vector_db.as_retriever(search_kwargs={"k": 2})
retriever = EnsembleRetriever(retrievers=[bm25_retriever, semantic_retriever], weights=[0.5, 0.5])
logging.info("¡NotarIA está lista!")


# --- DEFINICIÓN DE LA ARQUITECTURA CONVERSACIONAL ---

# Se mantiene el prompt funcional anterior
contextualize_q_system_prompt = """Dada una conversación y una última pregunta del usuario, reformula la última pregunta para que sea una pregunta autocontenida, en su idioma original. La pregunta debe tener todo el contexto necesario para ser entendida por sí misma."""
contextualize_q_prompt = ChatPromptTemplate.from_messages([("system", contextualize_q_system_prompt), MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}")])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

qa_system_prompt = """Eres NotarIA, un asistente legal de alta precisión. Tu única fuente de verdad es el CONTEXTO que se te proporciona a continuación y no utilizar nada de conocimientos que esten fuera del promtp de contexto. Está estrictamente prohibido usar cualquier conocimiento externo o pre-entrenado. Responde la PREGUNTA del usuario basándote exclusivamente en el CONTEXTO. Si la respuesta se encuentra en el CONTEXTO, al final de tu respuesta, DEBES citar tus fuentes incluyendo el nombre del archivo y el número de página que se encuentran en los metadatos. Por ejemplo: (Fuente: constitucion.pdf, página 15). Si la información no se encuentra en el CONTEXTO, debes responder EXACTAMENTE y únicamente con la frase: 'La información solicitada no se encuentra en los documentos que he procesado.'.

CONTEXTO:
{context}"""
qa_prompt = ChatPromptTemplate.from_messages([("system", qa_system_prompt), MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}")])

Youtube_chain = create_stuff_documents_chain(llm, qa_prompt)
conversational_rag_chain = create_retrieval_chain(history_aware_retriever, Youtube_chain)


# --- GESTIÓN DE LA MEMORIA DE LA CONVERSACIÓN ---
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store: 
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# --- FUNCIÓN PRINCIPAL DEL CHATBOT ---
def responder(message, history, session_id="my_session"):
    chat_history = get_session_history(session_id)
    logging.info(f"\n>>>>>>>>>>>>>>>>> NUEVA CONSULTA <<<<<<<<<<<<<<<<<")
    logging.info(f"Mensaje recibido: '{message}'")
    
    response = conversational_rag_chain.invoke({"input": message, "chat_history": chat_history.messages})
    
    logging.info("\n--- CONTEXTO RECUPERADO ---")
    if "context" in response and response["context"]:
        for i, doc in enumerate(response["context"]):
            logging.info(f"  Chunk {i+1} (Fuente: {doc.metadata.get('source', 'N/A')}, Pág: {doc.metadata.get('page', 'N/A')})")
            logging.info(f"  Contenido: {doc.page_content[:250]}...")
    else:
        logging.info("  No se recuperó ningún contexto.")
    logging.info("---------------------------\n")

    logging.info(f"Respuesta generada: {response['answer']}")
    
    chat_history.add_user_message(message)
    chat_history.add_ai_message(response["answer"])
    
    return response["answer"]

import subprocess

# --- ACTUALIZAR BASE VECTORIAL DESDE ARCHIVOS SUBIDOS ---
def actualizar_vectordb_archivos(files):
    """
    Copia los archivos subidos a la carpeta ./documentos_fuente,
    ejecuta la reconstrucción de la base vectorial y recarga Chroma.
    """
    import shutil
    import subprocess
    global vector_db, retriever

    if not files:
        return "No se cargaron archivos nuevos."

    try:
        # 1. Guardar los archivos en ./documentos_fuente
        destino = "./documentos_fuente"
        os.makedirs(destino, exist_ok=True)
        nombres = []
        for f in files:
            nombre_destino = os.path.join(destino, os.path.basename(f.name))
            shutil.copy(f.name, nombre_destino)
            nombres.append(os.path.basename(f.name))

        
        
        # 2. Ejecutar el script de creación de la base vectorial
        script_path = os.path.join(os.path.dirname(__file__), "crear_vectordb_general.py")
        resultado = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(script_path),
            capture_output=True, text=True
        )

        logging.info("---- STDOUT crear_vectordb_general.py ----")
        logging.info(resultado.stdout)
        logging.error(resultado.stderr)
        logging.info("------------------------------------------------")

        if resultado.returncode == 0:
            persist_dir = os.path.abspath("./chroma_db_final")
            embeddings = OllamaEmbeddings(model="mxbai-embed-large")
            vector_db = Chroma(persist_directory=persist_dir,
                            embedding_function=embeddings)
            semantic_retriever = vector_db.as_retriever(search_kwargs={"k": 2})
            retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, semantic_retriever],
                weights=[0.5, 0.5]
            )
            logging.info("Base vectorial actualizada y recargada correctamente.")
            return f"{len(nombres)} archivo(s) agregado(s) y base vectorial actualizada con éxito."
        else:
            logging.error(f"Error al regenerar la base vectorial: {resultado.stderr}")
            return f"Error en la actualización:\n{resultado.stderr}"

        

    except Exception as e:
        logging.error(f"Error durante la carga de archivos: {e}")
        return f"Error inesperado: {e}"
    
    
# --- INTERFAZ CON PANEL LATERAL DE CARGA ---
with gr.Blocks(theme=theme, css=css_styles, title="NotarIA 🧠 (Búsqueda Híbrida)") as interfaz:
    gr.Markdown("## NotarIA — Asistente Legal Inteligente")

    with gr.Row(equal_height=True):
        # --- COLUMNA IZQUIERDA: SUBIR DOCUMENTOS ---
        with gr.Column(scale=0.4):
            gr.Markdown("### Cargar nuevos documentos")
            file_upload = gr.File(
                label="Arrastra o selecciona tus archivos (.pdf o .txt)",
                file_count="multiple"
            )
            salida_carga = gr.Textbox(
                label="Estado de actualización",
                placeholder="Aquí se mostrará el resultado...",
                lines=4,
                interactive=False
            )
            # Cuando se cargan archivos → llama a la función actualizar_vectordb_archivos
            file_upload.upload(fn=actualizar_vectordb_archivos,
                               inputs=file_upload,
                               outputs=salida_carga)

        # --- COLUMNA DERECHA: CHAT PRINCIPAL ---
        with gr.Column(scale=0.6):
            chat = gr.ChatInterface(
                fn=responder,
                title=None,
                description="Asistente legal para consulta de documentos. Potenciado por un sistema de búsqueda híbrida.",
                chatbot=gr.Chatbot(
                    height=500,
                    type="messages",
                    avatar_images=("./user_avatar.png", "./bot_avatar.png")
                ),
                examples=[
                    ["¿Quién fue el cliente comprador en el expediente 2024-03-02B?"],
                    ["Soy un ciudadano extranjero y quiero comprar una casa en la playa, en Cancún."],
                    ["Si una persona es detenida, ¿cuáles son sus derechos mínimos durante el proceso?"]
                ]
            )


# --- LANZAMIENTO DE LA APP ---
if __name__ == "__main__":
    interfaz.launch()