import psycopg

# Abre una conexión con la base de datos PostgreSQL local.
connection = psycopg.connect(
    host="localhost",
    port=5432,
    dbname="dogs",
    user="postgres",
    password="postgres"
)

print("Conexión a PostgreSQL correcta")

# Cierra la conexión al terminar la prueba.
connection.close()