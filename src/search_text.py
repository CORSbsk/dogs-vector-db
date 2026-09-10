import open_clip
import torch
import psycopg


# --------------------------------------------------
# Configuración
# --------------------------------------------------

QUERY = "brown dog laying on the floor"

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "dogs",
    "user": "postgres",
    "password": "postgres",
}


# --------------------------------------------------
# Cargar CLIP
# --------------------------------------------------

print("Cargando CLIP...")

model, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")

model.eval()


# --------------------------------------------------
# Convertir texto a embedding
# --------------------------------------------------

print(f"Consulta: {QUERY}")

text = tokenizer([QUERY])

with torch.no_grad():
    embedding = model.encode_text(text)

embedding = embedding / embedding.norm(
    dim=-1,
    keepdim=True
)

vector = embedding[0].tolist()


# --------------------------------------------------
# Buscar en PostgreSQL
# --------------------------------------------------

connection = psycopg.connect(**DB_CONFIG)

with connection.cursor() as cursor:

    cursor.execute(
        """
        SELECT
            id,
            breed,
            filename,
            image_path,
            embedding <=> %s AS distance
        FROM dogs
        ORDER BY embedding <=> %s
        LIMIT 5;
        """,
        (str(vector), str(vector))
    )

    results = cursor.fetchall()

connection.close()


# --------------------------------------------------
# Mostrar resultados
# --------------------------------------------------

print()
print("RESULTADOS")
print("=" * 60)

for position, result in enumerate(results, start=1):

    id_, breed, filename, image_path, distance = result

    print(f"{position}. {filename}")
    print(f"   Raza: {breed}")
    print(f"   Distancia: {distance:.4f}")
    print(f"   Ruta: {image_path}")
    print()