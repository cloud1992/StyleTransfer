import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


# Función para cargar las imágenes
def load_image(image_path):
    img = Image.open(image_path)
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = transform(img).unsqueeze(0)
    return img


# Función para guardar la imagen procesada
def save_image(img, output_path):
    img = img.squeeze(0).detach().numpy()
    img = img.transpose(1, 2, 0)
    img = (img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    img = np.clip(img, 0, 1) * 255
    img = img.astype("uint8")
    Image.fromarray(img).save(output_path)


# Función para aplicar style transfer
def apply_style_transfer(content_path, style_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar las imágenes
    # content_image = load_image(content_path).to(device)
    # style_image = load_image(style_path).to(device)
    # input_image = content_image.clone().to(device).requires_grad_(True)

    # Cargar el modelo de red neuronal
    cnn = torch.hub.load("pytorch/vision:v0.6.0", "vgg19", pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # Definir las capas de contenido y estilo
    content_layers = ["conv4_2"]
    style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

    # Calcular las matrices de estilo y contenido
    content_features = {}
    style_features = {}

    def get_features(module, input, output):
        if isinstance(module, torch.nn.Conv2d):
            name = f"conv{module.bias.numel()}"
            if name in content_layers:
                content_features[name] = output.detach()
            if name in style_layers:
                style_gram = torch.nn.functional.conv2d(output, output, groups=output.shape[1])
                style_features[name] = style_gram.detach()

    cnn.register_forward_hook(get_features)

    # Normalizar las imágenes de entrada
    def normalize(image):
        return (image - cnn_normalization_mean.view(1, -1, 1, 1)) / cnn_normalization_std.view(1, -1, 1, 1)

    # Definir la función de pérdida
    def content_loss(input_features, target_features):
        return torch.mean((input_features - target_features) ** 2)

    def style_loss(input_gram, target_gram):
        _, c, h, w = input_gram.shape
        # input_features = input_gram.view(c, h * w)
        # target_features = target_gram.view(c, h * w)
        # gram_input = torch.mm
