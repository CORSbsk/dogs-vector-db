from pathlib import Path
import random
import shutil

# Configuración
SOURCE = Path("dataset/Images")
DESTINATION = Path("dataset/sample")

NUM_BREEDS = 10
IMAGES_PER_BREED = 20

random.seed(42)

# Crear destino
DESTINATION.mkdir(parents=True, exist_ok=True)

# Obtener las razas
breeds = sorted([
    folder for folder in SOURCE.iterdir()
    if folder.is_dir()
])

# Seleccionar 10 razas
selected_breeds = random.sample(
    breeds,
    min(NUM_BREEDS, len(breeds))
)

print("Razas seleccionadas:")

for breed in selected_breeds:
    images = [
        image for image in breed.iterdir()
        if image.is_file()
    ]

    selected_images = random.sample(
        images,
        min(IMAGES_PER_BREED, len(images))
    )

    breed_destination = DESTINATION / breed.name
    breed_destination.mkdir(parents=True, exist_ok=True)

    print(f"{breed.name}: {len(selected_images)} imágenes")

    for image in selected_images:
        shutil.copy2(
            image,
            breed_destination / image.name
        )

print("\nMuestra creada correctamente.")