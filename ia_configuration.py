import pickle

import faiss
import google.generativeai as genai
import numpy as np
import streamlit as st
from datasets import load_dataset
from google import genai
from langchain.schema import HumanMessage
from sentence_transformers import SentenceTransformer

# API de Gemini
client = genai.Client(api_key="AIzaSyATKWU248g389QPbpssNfKup0bYZnaN_8Y")  # API key


# Cargar el dataset de Wikipedia
@st.cache_resource
def load_wikipedia_dataset():
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.es",
        split="train[:1000]",  # Cargar solo 1000 documentos para prueba
    )
    return dataset


# Preprocesamiento del texto
def preprocess_text(text):
    # Limpieza básica: eliminar saltos de línea y espacios extra
    text = " ".join(text.split())
    return text


# Guardar y cargar embeddings para evitar recalcularlos
def save_embeddings(embeddings, filename="embeddings.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(embeddings, f)


def load_embeddings(filename="embeddings.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)


# Generar embeddings y crear índice FAISS
@st.cache_resource
def create_faiss_index(_dataset):
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    texts = [preprocess_text(doc["text"]) for doc in _dataset]

    # Revisar si ya tenemos los embeddings calculados
    try:
        embeddings = load_embeddings()  # Cargar los embeddings guardados
    except FileNotFoundError:
        embeddings = model.encode(texts, show_progress_bar=True)
        save_embeddings(embeddings)  # Guardar embeddings después de generarlos

    # Crear índice FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
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


# Generar respuesta usando Gemini
def generate_response(query, relevant_docs):
    try:
        # Verificar si hay documentos relevantes
        if not relevant_docs:
            return "No se encontró información relevante para responder a tu pregunta."

        # Crear el contexto
        context = "\n".join(relevant_docs)
        prompt = f"Basado en la siguiente información:\n{context}\n\nResponde de manera clara y concisa a la pregunta: {query}"

        # Generar la respuesta con Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
        )

        # Devolver la respuesta generada
        return response.text if response.text else "No se pudo generar una respuesta."

    except Exception as e:
        return f"Error al generar la respuesta: {str(e)}"
