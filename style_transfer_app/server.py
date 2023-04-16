from sanic import Sanic, response
from sanic.exceptions import abort

from .common.utils import apply_style_transfer

app = Sanic(__name__)


# Función para manejar la solicitud del usuario
@app.get("/")
async def handle_request(request):
    try:
        with open("static/index.html", "r") as f:
            html_content = f.read()
        return response.html(html_content)
    except FileNotFoundError:
        abort(404)


# Función para procesar la imagen y aplicar el estilo
@app.post("/process_image")
async def process_image(request):
    # Obtener las imágenes del formulario
    content_image = request.files.get("content_image")
    style_image = request.files.get("style_image")

    # Guardar las imágenes en el servidor
    content_path = "/tmp/content.jpg"
    style_path = "/tmp/style.jpg"
    content_image.save(content_path)
    style_image.save(style_path)

    # Aplicar style transfer
    output_path = "/tmp/output.jpg"
    apply_style_transfer(content_path, style_path, output_path)

    # Devolver la imagen procesada al usuario
    return await response.file(output_path, filename="output.jpg")
