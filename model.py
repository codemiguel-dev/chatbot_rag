import sqlite3


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
