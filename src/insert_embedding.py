from pathlib import Path

import torch
import open_clip
import psycopg
from PIL import Image


# --------------------------------------------------
# 1. Cargar CLIP
# --------------------------------------------------

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

model.eval()


# --------------------------------------------------
# 2. Buscar una imagen
# --------------------------------------------------

sample_dir = Path("dataset/sample")

images = list(sample_dir.rglob("*.jpg"))

if not images:
    raise RuntimeError("No se encontraron imágenes.")

image_path = images[0]

print(f"Procesando: {image_path}")


# --------------------------------------------------
# 3. Generar embedding
# --------------------------------------------------

image = preprocess(
    Image.open(image_path).convert("RGB")
).unsqueeze(0)

with torch.no_grad():
    embedding = model.encode_image(image)

embedding = embedding / embedding.norm(dim=-1, keepdim=True)

vector = embedding[0].tolist()

print(f"Vector generado: {len(vector)} dimensiones")


# --------------------------------------------------
# 4. Obtener información de la imagen
# --------------------------------------------------

breed = image_path.parent.name
filename = image_path.name


# --------------------------------------------------
# 5. Conectar a PostgreSQL
# --------------------------------------------------

connection = psycopg.connect(
    host="localhost",
    port=5432,
    dbname="dogs",
    user="postgres",
    password="postgres"
)


# --------------------------------------------------
# 6. Insertar vector
# --------------------------------------------------

with connection.cursor() as cursor:

    cursor.execute(
        """
        INSERT INTO dogs (
            breed,
            filename,
            image_path,
            embedding
        )
        VALUES (%s, %s, %s, %s)
        """,
        (
            breed,
            filename,
            str(image_path),
            vector
        )
    )

connection.commit()
connection.close()

print("Embedding guardado correctamente en PostgreSQL.")