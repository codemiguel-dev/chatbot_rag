import sqlite3

import faiss
import google.generativeai as genai
import numpy as np
import streamlit as st
from datasets import load_dataset
from google.api_core import retry
from sentence_transformers import SentenceTransformer

# Configuración de la API de Gemini
genai.configure(
    api_key="AIzaSyATKWU248g389QPbpssNfKup0bYZnaN_8Y"
)  # Reemplaza con tu clave API


# 🔹 Conectar a SQLite (para guardar historial de conversaciones)
def inicializar_bd():
    conn = sqlite3.connect("database/historial_chat.db")
    cursor = conn.cursor()
    # Crear la tabla si no existe
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS historial (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT NOT NULL,
            usuario TEXT NOT NULL,
            chatbot TEXT NOT NULL
        )
    """
    )
    conn.commit()
    conn.close()


def guardar_historial(usuario, chatbot):
    conn = sqlite3.connect("database/historial_chat.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO historial (fecha, usuario, chatbot) VALUES (datetime('now'), ?, ?)",
        (usuario, chatbot),
    )
    conn.commit()
    conn.close()


def obtener_historial():
    conn = sqlite3.connect("database/historial_chat.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT fecha, usuario, chatbot FROM historial ORDER BY id DESC LIMIT 10"
    )
    historial = cursor.fetchall()
    conn.close()
    return historial


# Cargar el dataset de Wikipedia
@st.cache_resource
def load_wikipedia_dataset():
    dataset = load_dataset(
        "wikimedia/wikipedia", "20231101.es", split="train[:1000]"
    )  # Cargar solo 1000 documentos para prueba
    return dataset


# Preprocesamiento del texto
def preprocess_text(text):
    # Limpieza básica: eliminar saltos de línea y espacios extra
    text = " ".join(text.split())
    return text


# Generar embeddings y crear índice FAISS
@st.cache_resource
def create_faiss_index(_dataset):  # Cambia 'dataset' a '_dataset'
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )  # Modelo multilingüe
    texts = [preprocess_text(doc["text"]) for doc in _dataset]  # Usa '_dataset' aquí
    embeddings = model.encode(texts, show_progress_bar=True)

    # Crear índice FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Índice basado en distancia L2
    index.add(np.array(embeddings).astype("float32"))
    return index, model, texts


# Recuperar documentos relevantes
def retrieve_documents(query, index, model, texts, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(
        np.array(query_embedding).astype("float32"), top_k
    )
    relevant_docs = [texts[i] for i in indices[0]]
    return relevant_docs


@retry.Retry(
    initial=1.0,  # Tiempo inicial de espera (1 segundo)
    maximum=10.0,  # Tiempo máximo de espera (10 segundos)
    multiplier=2.0,  # Factor de multiplicación para el tiempo de espera
    deadline=30.0,  # Tiempo máximo total para reintentos (30 segundos)
)

# Generar respuesta usando Gemini
def generate_response(query, relevant_docs):
    context = "\n".join(relevant_docs)
    prompt = f"Basado en la siguiente información:\n{context}\n\nResponde a la pregunta: {query}"
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text


# Interfaz de usuario con Streamlit
def main():
    st.title("🤖 Chatbot con Wikipedia y Gemini")
    st.write(
        "Este sistema permite hacer preguntas y obtener respuestas basadas en artículos de Wikipedia."
    )

    # Inicializar la base de datos
    inicializar_bd()

    # Cargar dataset y crear índice FAISS
    dataset = load_wikipedia_dataset()
    index, model, texts = create_faiss_index(dataset)

    # Entrada de usuario
    query = st.text_input("Haz tu pregunta:")

    # Botón para responder
    if st.button("Responder"):
        if query:
            with st.spinner("Buscando información relevante..."):
                # Recuperar documentos relevantes
                relevant_docs = retrieve_documents(query, index, model, texts)

                # Generar respuesta usando Gemini
                response = generate_response(query, relevant_docs)

                # Mostrar resultados
                st.subheader("🤖 **Gemini dice:**")
                st.write(response)

                st.subheader("Documentos relevantes:")
                for i, doc in enumerate(relevant_docs):
                    st.write(f"**Documento {i+1}:**")
                    st.write(
                        doc[:500] + "..."
                    )  # Mostrar solo un fragmento del documento

                # Guardar en el historial
                guardar_historial(query, response)
                st.success("Respuesta guardada en el historial.")
        else:
            st.warning("Por favor, escribe una pregunta.")

    # 🔹 Mostrar historial de conversación en el sidebar
    st.sidebar.subheader("📜 Historial de Conversación")
    historial = obtener_historial()
    for fecha, usuario, chatbot in historial:
        st.sidebar.write(f"📅 {fecha}")
        st.sidebar.write(f"👤 **Tú:** {usuario}")
        st.sidebar.write(f"🤖 **Gemini:** {chatbot}")
        st.sidebar.write("---")


if __name__ == "__main__":
    main()
