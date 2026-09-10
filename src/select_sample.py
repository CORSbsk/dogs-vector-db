from pathlib import Path
import random
import shutil

# Rutas y cantidad de datos que se copiarán.
SOURCE = Path("dataset/Images")
DESTINATION = Path("dataset/sample")

NUM_BREEDS = 10
IMAGES_PER_BREED = 20

random.seed(42)

# Crea la carpeta de destino si todavía no existe.
DESTINATION.mkdir(parents=True, exist_ok=True)

# Obtiene las carpetas, donde cada una representa una raza.
breeds = sorted([
    folder for folder in SOURCE.iterdir()
    if folder.is_dir()
])

# Elige razas al azar sin superar la cantidad disponible.
selected_breeds = random.sample(
    breeds,
    min(NUM_BREEDS, len(breeds))
)

print("Razas seleccionadas:")

for breed in selected_breeds:
    # Busca los archivos de imagen de la raza actual.
    images = [
        image for image in breed.iterdir()
        if image.is_file()
    ]

    # Elige imágenes al azar sin superar las disponibles.
    selected_images = random.sample(
        images,
        min(IMAGES_PER_BREED, len(images))
    )

    breed_destination = DESTINATION / breed.name
    breed_destination.mkdir(parents=True, exist_ok=True)

    print(f"{breed.name}: {len(selected_images)} imágenes")

    for image in selected_images:
        # Copia la imagen y conserva sus datos originales.
        shutil.copy2(
            image,
            breed_destination / image.name
        )

print("\nMuestra creada correctamente.")