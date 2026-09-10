import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import torch
import open_clip
import psycopg
from PIL import Image, ImageTk


# ============================================================
# CONFIGURACIÓN
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "dogs",
    "user": "postgres",
    "password": "postgres",
}

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

RESULTS_LIMIT = 5

# Tamaño de las miniaturas
IMAGE_WIDTH = 180
IMAGE_HEIGHT = 140


# ============================================================
# CARGAR CLIP
# ============================================================

print("Cargando CLIP...")

model, _, _ = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED
)

tokenizer = open_clip.get_tokenizer(MODEL_NAME)

model.eval()

print("CLIP cargado correctamente.")


# ============================================================
# FUNCIÓN DE BÚSQUEDA
# ============================================================

def search_database(query):
    """
    Convierte el texto en un embedding y busca
    las 5 imágenes más similares en pgvector.
    """

    # --------------------------------------------
    # Texto -> embedding
    # --------------------------------------------

    text = tokenizer([query])

    with torch.no_grad():
        embedding = model.encode_text(text)

    embedding = embedding / embedding.norm(
        dim=-1,
        keepdim=True
    )

    vector = embedding[0].tolist()

    # --------------------------------------------
    # PostgreSQL
    # --------------------------------------------

    connection = psycopg.connect(**DB_CONFIG)

    try:

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
                LIMIT %s;
                """,
                (
                    str(vector),
                    str(vector),
                    RESULTS_LIMIT
                )
            )

            results = cursor.fetchall()

    finally:

        connection.close()

    return results


# ============================================================
# MOSTRAR RESULTADOS
# ============================================================

def show_results(results):

    # Eliminar resultados anteriores
    for widget in results_frame.winfo_children():
        widget.destroy()

    # Mantener referencias a las imágenes
    image_references.clear()

    if not results:
        ttk.Label(
            results_frame,
            text="No se encontraron resultados."
        ).pack(pady=20)

        return

    for position, result in enumerate(results, start=1):

        id_, breed, filename, image_path, distance = result

        # --------------------------------------------
        # Contenedor de cada resultado
        # --------------------------------------------

        result_frame = ttk.Frame(
            results_frame,
            padding=10,
            relief="solid"
        )

        result_frame.pack(
            fill="x",
            padx=10,
            pady=5
        )

        # --------------------------------------------
        # Imagen
        # --------------------------------------------

        image_label = ttk.Label(result_frame)

        image_label.pack(
            side="left",
            padx=(0, 15)
        )

        try:

            path = Path(image_path)

            # Si la ruta no existe, intentar desde el directorio
            # raíz del proyecto.
            if not path.exists():
                path = Path.cwd() / image_path

            image = Image.open(path)

            image.thumbnail(
                (IMAGE_WIDTH, IMAGE_HEIGHT)
            )

            photo = ImageTk.PhotoImage(image)

            image_label.configure(
                image=photo
            )

            # IMPORTANTE:
            # Tkinter necesita conservar una referencia.
            image_references.append(photo)

        except Exception as error:

            image_label.configure(
                text="Imagen no disponible"
            )

            print(
                f"No se pudo cargar {image_path}: {error}"
            )

        # --------------------------------------------
        # Información
        # --------------------------------------------

        info_frame = ttk.Frame(result_frame)

        info_frame.pack(
            side="left",
            fill="both",
            expand=True
        )

        ttk.Label(
            info_frame,
            text=f"Resultado #{position}",
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w")

        ttk.Label(
            info_frame,
            text=f"ID: {id_}"
        ).pack(anchor="w")

        ttk.Label(
            info_frame,
            text=f"Raza: {breed}"
        ).pack(anchor="w")

        ttk.Label(
            info_frame,
            text=f"Archivo: {filename}"
        ).pack(anchor="w")

        ttk.Label(
            info_frame,
            text=f"Distancia: {distance:.4f}"
        ).pack(anchor="w")

        ttk.Label(
            info_frame,
            text=f"Ruta: {image_path}"
        ).pack(anchor="w")


# ============================================================
# EJECUTAR BÚSQUEDA
# ============================================================

def execute_search():

    query = query_entry.get().strip()

    if not query:

        messagebox.showwarning(
            "Consulta vacía",
            "Escribe una descripción del perro."
        )

        return

    status_label.config(
        text="Buscando..."
    )

    search_button.config(
        state="disabled"
    )

    root.update_idletasks()

    try:

        results = search_database(query)

        show_results(results)

        status_label.config(
            text=f"{len(results)} resultados encontrados"
        )

    except Exception as error:

        messagebox.showerror(
            "Error",
            f"No se pudo realizar la búsqueda:\n\n{error}"
        )

        status_label.config(
            text="Error durante la búsqueda"
        )

    finally:

        search_button.config(
            state="normal"
        )


# ============================================================
# ENTER = BUSCAR
# ============================================================

def on_enter(event):
    execute_search()


# ============================================================
# VENTANA PRINCIPAL
# ============================================================

root = tk.Tk()

root.title(
    "Dog Vector Search"
)

root.geometry(
    "950x800"
)

root.minsize(
    800,
    600
)


# ============================================================
# ESTILO
# ============================================================

style = ttk.Style()

try:
    style.theme_use("vista")
except:
    pass


# ============================================================
# ENCABEZADO
# ============================================================

header_frame = ttk.Frame(
    root,
    padding=20
)

header_frame.pack(
    fill="x"
)

title_label = ttk.Label(
    header_frame,
    text="Dog Vector Search",
    font=("Segoe UI", 20, "bold")
)

title_label.pack(
    anchor="w"
)

description_label = ttk.Label(
    header_frame,
    text="Búsqueda semántica de imágenes mediante CLIP + pgvector"
)

description_label.pack(
    anchor="w",
    pady=(5, 15)
)


# ============================================================
# BUSCADOR
# ============================================================

search_frame = ttk.Frame(
    root,
    padding=(20, 0, 20, 10)
)

search_frame.pack(
    fill="x"
)

query_entry = ttk.Entry(
    search_frame,
    font=("Segoe UI", 12)
)

query_entry.pack(
    side="left",
    fill="x",
    expand=True,
    ipady=7
)

query_entry.insert(
    0,
    "a brown dog laying on the floor"
)

search_button = ttk.Button(
    search_frame,
    text="Buscar",
    command=execute_search
)

search_button.pack(
    side="left",
    padx=(10, 0),
    ipadx=15,
    ipady=5
)

query_entry.bind(
    "<Return>",
    on_enter
)


# ============================================================
# ESTADO
# ============================================================

status_label = ttk.Label(
    root,
    text="Escribe una consulta y presiona Buscar",
    padding=(20, 5)
)

status_label.pack(
    fill="x"
)


# ============================================================
# CONTENEDOR CON SCROLL
# ============================================================

container = ttk.Frame(
    root
)

container.pack(
    fill="both",
    expand=True,
    padx=20,
    pady=10
)

canvas = tk.Canvas(
    container
)

scrollbar = ttk.Scrollbar(
    container,
    orient="vertical",
    command=canvas.yview
)

results_frame = ttk.Frame(
    canvas
)

results_frame.bind(
    "<Configure>",
    lambda event: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas_window = canvas.create_window(
    (0, 0),
    window=results_frame,
    anchor="nw"
)

canvas.configure(
    yscrollcommand=scrollbar.set
)

canvas.pack(
    side="left",
    fill="both",
    expand=True
)

scrollbar.pack(
    side="right",
    fill="y"
)


# Hacer que el contenido ocupe todo el ancho
def resize_results(event):

    canvas.itemconfig(
        canvas_window,
        width=event.width
    )


canvas.bind(
    "<Configure>",
    resize_results
)


# ============================================================
# REFERENCIAS DE IMÁGENES
# ============================================================

image_references = []


# ============================================================
# INICIAR
# ============================================================

root.mainloop()