# import io
import logging

# from common.utils import apply_style_transfer
# from PIL import Image
from sanic import Sanic, response
from sanic.exceptions import HTTPException

# import os

# import numpy as np


logging.basicConfig(level=logging.INFO)
app = Sanic(__name__)


# Función para manejar la solicitud del usuario
@app.get("/")
async def handle_request(request):
    try:
        with open("static/index.html", "r") as f:
            html_content = f.read()
        return response.html(html_content)
    except FileNotFoundError:
        HTTPException("Página no encontrada")
        with open("static/index.html", "r") as f:
            html_content = f.read()
            print(html_content)
        return response.text("Página no encontrada")


# Función para procesar la imagen y aplicar el estilo
# @app.post("/process")
# async def process(request):
#     # Obtener las imágenes del formulario
#     content_image = request.files.get("content_image")
#     style_image = request.files.get("style_image")

#     # content_image_name = content_image.name
#     # style_image_name = style_image.name

#     content_image = content_image.body
#     style_image = style_image.body

#     # Abrir la imagen usando PIL
#     pil_style_image = Image.open(io.BytesIO(style_image))
#     pil_content_image = Image.open(io.BytesIO(content_image))

#     # Convertir la imagen PIL a un array de NumPy
#     np_image = np.array(pil_style_image)
#     np_content_image = np.array(pil_content_image)
#     logging.info(f"Content image type: {type(np_image)}")
#     logging.info(f"Content image type: {type(np_content_image)}")

#     # Guardar las imágenes en el servidor
#     content_path = os.path.join("tmp", "content.jpg")
#     style_path = os.path.join("tmp", "style.jpg")
#     pil_content_image.save(content_path)
#     pil_style_image.save(style_path)

#     # Aplicar style transfer
#     apply_style_transfer(pil_content_image, pil_style_image)

#     # Devolver la imagen procesada al usuario
#     return response.text("todo ok11")
#     # return await response.file(output_path, filename="output.jpg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
