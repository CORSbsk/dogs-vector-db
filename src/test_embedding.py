from pathlib import Path

import torch
import open_clip
from PIL import Image


# Carga el modelo CLIP y la función que prepara las imágenes.
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

model.eval()

# Busca imágenes dentro de todas las carpetas de la muestra.
sample_dir = Path("dataset/sample")

images = list(sample_dir.rglob("*.jpg"))

if not images:
    raise RuntimeError("No se encontraron imágenes .jpg en dataset/sample")

image_path = images[0]

print(f"Imagen: {image_path}")

# Convierte la imagen al formato que espera CLIP.
image = preprocess(
    Image.open(image_path).convert("RGB")
).unsqueeze(0)

# Genera el vector numérico que representa la imagen.
with torch.no_grad():
    embedding = model.encode_image(image)

# Normaliza el vector para facilitar la comparación entre imágenes.
embedding = embedding / embedding.norm(dim=-1, keepdim=True)

vector = embedding[0].tolist()

print(f"Dimensiones: {len(vector)}")
print(f"Primeros 10 valores: {vector[:10]}")