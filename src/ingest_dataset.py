from pathlib import Path

import torch
import open_clip
import psycopg
from PIL import Image


# --------------------------------------------------
# Configuración
# --------------------------------------------------

DATASET_DIR = Path("dataset/sample")

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "dogs",
    "user": "postgres",
    "password": "postgres",
}


# --------------------------------------------------
# Cargar modelo CLIP
# --------------------------------------------------

print("Cargando CLIP...")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

model.eval()

print("CLIP cargado.")


# --------------------------------------------------
# Obtener imágenes
# --------------------------------------------------

images = list(DATASET_DIR.rglob("*.jpg"))

print(f"Imágenes encontradas: {len(images)}")


# --------------------------------------------------
# Conectar a PostgreSQL
# --------------------------------------------------

connection = psycopg.connect(**DB_CONFIG)


# --------------------------------------------------
# Procesar imágenes
# --------------------------------------------------

with connection.cursor() as cursor:

    for index, image_path in enumerate(images, start=1):

        breed = image_path.parent.name
        filename = image_path.name

        # Evitar duplicados
        cursor.execute(
            """
            SELECT id
            FROM dogs
            WHERE image_path = %s
            """,
            (str(image_path),)
        )

        if cursor.fetchone():
            print(f"[{index}/{len(images)}] Ya existe: {filename}")
            continue

        try:

            # Abrir imagen
            image = Image.open(image_path).convert("RGB")

            # Preprocesar
            image_tensor = preprocess(image).unsqueeze(0)

            # Generar embedding
            with torch.no_grad():
                embedding = model.encode_image(image_tensor)

            # Normalizar
            embedding = embedding / embedding.norm(
                dim=-1,
                keepdim=True
            )

            vector = embedding[0].tolist()

            # Insertar
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

            print(
                f"[{index}/{len(images)}] "
                f"Insertada: {breed}/{filename}"
            )

        except Exception as error:

            print(
                f"[{index}/{len(images)}] "
                f"ERROR: {image_path}"
            )

            print(error)


# --------------------------------------------------
# Confirmar cambios
# --------------------------------------------------

connection.commit()
connection.close()

print()
print("Proceso terminado.")