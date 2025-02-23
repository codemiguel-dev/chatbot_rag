import streamlit as st
from google import genai
from google.api_core import retry

from conect_bd import inicializar_bd
from ia_configuration import (
    create_faiss_index,
    generate_response,
    load_wikipedia_dataset,
    retrieve_documents,
)
from model import guardar_historial, obtener_historial

# API de Geminis
client = genai.Client(api_key="AIzaSyATKWU248g389QPbpssNfKup0bYZnaN_8Y")  # API


@retry.Retry(
    initial=1.0,  # Tiempo inicial de espera (1 segundo)
    maximum=10.0,  # Tiempo máximo de espera (10 segundos)
    multiplier=2.0,  # Factor de multiplicación para el tiempo de espera
    deadline=30.0,  # Tiempo máximo total para reintentos (30 segundos)
)

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
                response = generate_response(query, relevant_docs, client)

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
