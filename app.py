import sqlite3

import faiss
import google.generativeai as genai
import numpy as np
import streamlit as st
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# 🔹 Configurar la API de Gemini
API_KEY = "AIzaSyATKWU248g389QPbpssNfKup0bYZnaN_8Y"  # Reemplázala con tu clave de API
genai.configure(api_key=API_KEY)

# 🔹 Cargar modelo de embeddings
modelo_embeddings = SentenceTransformer("all-MiniLM-L6-v2")


# 🔹 Conectar a SQLite (para guardar historial de conversaciones)
def inicializar_bd():
    conn = sqlite3.connect("historial_chat.db")
    cursor = conn.cursor()
    conn.commit()
    conn.close()


def guardar_historial(usuario, chatbot):
    conn = sqlite3.connect("historial_chat.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO historial (fecha, usuario, chatbot) VALUES (datetime('now'), ?, ?)",
        (usuario, chatbot),
    )
    conn.commit()
    conn.close()


def obtener_historial():
    conn = sqlite3.connect("historial_chat.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT fecha, usuario, chatbot FROM historial ORDER BY id DESC LIMIT 10"
    )
    historial = cursor.fetchall()
    conn.close()
    return historial


# 🔹 Cargar dataset de Wikipedia
@st.cache_data
def cargar_dataset():
    dataset = load_dataset("wikimedia/wikipedia", "20231101.es", split="train")
    return dataset["text"][
        :500
    ]  # Solo cargamos los primeros 500 artículos para el ejemplo


articulos = cargar_dataset()

# 🔹 Crear base de datos vectorial FAISS
dimension = 384  # Dimensión del modelo de embeddings
index = faiss.IndexFlatL2(dimension)


# 🔹 Generar embeddings y almacenarlos en FAISS
@st.cache_data
def indexar_wikipedia(articulos):
    embeddings = modelo_embeddings.encode(articulos)
    index.add(np.array(embeddings, dtype=np.float32))
    return embeddings


embeddings_wiki = indexar_wikipedia(articulos)


# 🔹 Función para buscar documentos relevantes en Wikipedia
def buscar_wikipedia(pregunta, top_k=3):
    pregunta_embedding = modelo_embeddings.encode([pregunta])
    distancias, indices = index.search(
        np.array(pregunta_embedding, dtype=np.float32), k=top_k
    )
    resultados = [articulos[i] for i in indices[0]]
    return resultados


# 🔹 Función para generar respuesta con Gemini
def obtener_respuesta(pregunta):
    documentos = buscar_wikipedia(pregunta)
    contexto = "\n\n".join(documentos)
    model = genai.GenerativeModel("gemini-pro")
    respuesta = model.generate_content(
        f"Con base en esta información: {contexto}, responde la siguiente pregunta: {pregunta}"
    )
    return respuesta.text


# 🔹 Interfaz en Streamlit
st.title("🤖 Chatbot con Wikipedia y Gemini 2.0")

pregunta = st.text_input("Escribe tu pregunta:")

if st.button("Responder"):
    if pregunta:
        respuesta = obtener_respuesta(pregunta)
        guardar_historial(pregunta, respuesta)
        st.write("🤖 **Gemini dice:**", respuesta)
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


# Mostrar enlaces a los primeros 5 artículos
articulos = cargar_dataset()
st.write("Primeros 10 artículos de Wikipedia cargados:")

for idx, articulo in enumerate(articulos[:10]):
    # Crear un enlace para cada artículo
    link = f"[Artículo {idx+1}](#)"  # Puedes poner un enlace verdadero si tienes una URL válida
    st.markdown(
        f"{link}: {articulo[:1000]}..."
    )  # Muestra una parte del artículo con un enlace
