
# NotarIA 🧠 - Asistente Legal con Búsqueda Híbrida (RAG)

Este proyecto implementa un asistente legal conversacional llamado NotarIA. Utiliza un sistema de RAG (Retrieval-Augmented Generation) con búsqueda híbrida para responder preguntas basándose exclusivamente en un conjunto de documentos proporcionados por el usuario.

## Características Principales

* **Arquitectura RAG:** Genera respuestas a partir de la información recuperada de los documentos.
* **Búsqueda Híbrida:** Combina la búsqueda semántica (por significado) con la búsqueda por palabras clave (BM25) para obtener resultados más precisos.
* **Modelos Locales:** Utiliza Ollama para ejecutar modelos de lenguaje (`llama3`) y de embeddings (`mxbai-embed-large`) de forma local, garantizando la privacidad de los datos.
* **Interfaz Interactiva:** Provee una interfaz de chat amigable creada con Gradio.
* **Soporte Multiformato:** Procesa documentos en formato `.pdf` y `.txt`.

---

## 🚀 Guía de Instalación y Ejecución

Sigue estos pasos para poner en marcha el proyecto.

### Prerrequisitos

* Python 3.8 o superior.
* [Ollama](https://ollama.com/) instalado y en ejecución.

### Pasos

1.  **Clonar el Repositorio**
    ```bash
    git clone <https://github.com/obedvfc19/Proyecto_RAG.git>
    cd <nombre-de-la-carpeta-del-repositorio>
    ```

2.  **Crear y Activar un Entorno Virtual**
    Es una buena práctica aislar las dependencias del proyecto.
    ```bash
    # Crear el entorno
    python3 -m venv venv

    # Activar en Linux / macOS / WSL
    source venv/bin/activate
    ```

3.  **Instalar Dependencias**
    Instala todas las librerías necesarias desde el archivo `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Descargar los Modelos de Ollama**
    Abre otra terminal y ejecuta estos comandos para descargar los modelos necesarios.
    ```bash
    ollama pull llama3
    ollama pull mxbai-embed-large
    ```

---

## ⚙️ Uso del Programa

El proceso se divide en dos fases: crear la base de conocimiento y ejecutar la aplicación.

### Fase 1: Crear la Base de Datos Vectorial

1.  **Añade tus documentos:** Coloca todos tus archivos `.pdf` y `.txt` dentro de la carpeta `documentos_fuente/`.
2.  **Ejecuta el script de creación:** Asegúrate de que tu entorno virtual esté activado y ejecuta:
    ```bash
    python3 crear_vectordb_general.py
    ```
    Este proceso leerá tus documentos, los procesará y creará una carpeta `chroma_db_final` que contiene la base de datos vectorial.

### Fase 2: Ejecutar la Aplicación de Chat

1.  **Inicia la aplicación:** Con el entorno virtual aún activado, ejecuta el script principal.
    ```bash
    python3 app_notario.py
    ```
2.  **Abre la interfaz:** La terminal te proporcionará una URL local (ej. `http://127.0.0.1:7860`). Cópiala y pégala en tu navegador para comenzar a chatear con NotarIA
