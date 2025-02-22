import google.generativeai as genai
import streamlit as st
from datasets import load_dataset

# Configurar API de Gemini
API_KEY = "AIzaSyATKWU248g389QPbpssNfKup0bYZnaN_8Y"  # Reemplázala con tu clave API
genai.configure(api_key=API_KEY)


# Cargar el dataset de Wikipedia desde Hugging Face
@st.cache_data
def cargar_wikipedia():
    dataset = load_dataset(
        "wikimedia/wikipedia", "20231101.es", split="train"
    )  # Cargar en español
    return dataset


dataset = cargar_wikipedia()


# Función para buscar información en Wikipedia
def buscar_en_wikipedia(pregunta):
    resultados = [
        doc["text"]
        for doc in dataset.select(range(1000))
        if pregunta.lower() in doc["text"].lower()
    ]
    return resultados[:3]  # Retornar solo los 3 primeros resultados


# Función para obtener respuesta de Gemini usando Wikipedia
def obtener_respuesta(pregunta):
    contexto = buscar_en_wikipedia(pregunta)
    contexto_texto = (
        " ".join(contexto) if contexto else "No se encontró información en Wikipedia."
    )

    model = genai.GenerativeModel("gemini-pro")
    respuesta = model.generate_content(
        f"Basado en Wikipedia: {contexto_texto}\n\nPregunta: {pregunta}"
    )

    return respuesta.text


# Interfaz con Streamlit
st.title("Chatbot con Gemini 2.0 y Wikipedia")

pregunta = st.text_input("Escribe una pregunta:")

if st.button("Enviar"):
    if pregunta:
        respuesta = obtener_respuesta(pregunta)
        st.write("🤖 **Gemini dice:**", respuesta)
    else:
        st.warning("Por favor, escribe una pregunta.")
