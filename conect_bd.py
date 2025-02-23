import sqlite3


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
