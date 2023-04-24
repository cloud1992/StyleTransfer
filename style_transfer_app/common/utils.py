import logging

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import save_image

logging.basicConfig(level=logging.INFO)


# Función para aplicar style transfer
def apply_style_transfer(original_img, style_img):
    model = models.vgg19(pretrained=True).features

    class VGG(nn.Module):
        def __init__(self):
            super(VGG, self).__init__()
            self.select = ["0", "5", "10", "19", "28"]
            self.vgg = model

        def forward(self, x):
            features = []
            for name, layer in self.vgg._modules.items():
                x = layer(x)
                if name in self.select:
                    features.append(x)
            return features

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 356
    loader = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),  # scale imported image
            transforms.ToTensor(),  # transform it into a torch tensor
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  # normalize it
        ]
    )

    def load_image(image):
        image = loader(image).unsqueeze(0)
        return image.to(device)

    original_img = load_image(original_img)
    style_img = load_image(style_img)

    model = VGG().to(device).eval()

    generated = original_img.clone().requires_grad_(True)
    # Hyperparameters
    total_steps = 5000
    learning_rate = 0.001
    alpha = 1
    beta = 0.01
    optimizer = torch.optim.Adam([generated], lr=learning_rate)

    # training
    for step in range(total_steps):
        generated_features = model(generated)
        original_img_features = model(original_img)
        style_features = model(style_img)

        style_loss = original_loss = 0

        for gen_feature, orig_feature, style_feature in zip(generated_features, original_img_features, style_features):
            batch_size, channel, height, width = gen_feature.shape
            original_loss += torch.mean((gen_feature - orig_feature) ** 2)

            # compute gram matrix
            G = gen_feature.view(channel, height * width).mm(gen_feature.view(channel, height * width).t())
            A = style_feature.view(channel, height * width).mm(style_feature.view(channel, height * width).t())

            style_loss += torch.mean((G - A) ** 2)

        total_loss = alpha * original_loss + beta * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 5 == 0:
            print(total_loss)
            save_image(generated, "tmp/" + str(step) + ".jpg")
        print(step)
