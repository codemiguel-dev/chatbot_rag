# Usa una imagen base oficial de Python 3.9
FROM python:3.9-slim

# Instala las dependencias del sistema, incluyendo swig
RUN apt-get update && apt-get install -y \
    swig \
    build-essential \
    python3-dev

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo requirements.txt a la imagen
COPY requirements.txt .

# Instala las dependencias de Python desde el requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de tu aplicación al contenedor
COPY . .

# Expone el puerto en el que Streamlit va a correr
EXPOSE 8501

# Define el comando para ejecutar tu aplicación Streamlit
CMD ["streamlit", "run", "app_web.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
