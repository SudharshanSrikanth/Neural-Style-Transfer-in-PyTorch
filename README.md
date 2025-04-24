# Neural-Style-Transfer-in-PyTorch
# A deep learning project that fuses the content of one image with the artistic style of another using Neural Style Transfer. This implementation leverages a pre-trained VGG19 model to extract image features and applies style via loss functions computed directly on PyTorch tensors. PyTorch, VGG19, torchvision, matplotlib, PIL, LBFGS optimizer
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())

import torch.nn as nn
import torch.optim as optim

from torchvision import models, transforms
from PIL import Image
from matplotlib import pyplot as plt

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# --- Image Loader ---
def load_image(img_path, max_size=512):
    image = Image.open(img_path).convert('RGB')
    size = min(max(image.size), max_size)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)[:3, :, :].unsqueeze(0)
    return image.to(device)

# --- VGG Model Feature Extractor ---
class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features[:29].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = {}
        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '10': 'conv3_1', '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

# --- Gram Matrix ---
def gram_matrix(tensor):
    tensor = tensor.to(device)
    _, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    return torch.mm(features, features.t()) / (c * h * w)

# --- Load Images ---
content = load_image("me.jpeg")
print("Content image loaded")
style = load_image("starry_night.jpg")
print("Style image loaded")

# --- Initialize Target ---
target = content.clone().requires_grad_(True).to(device)

# --- Feature Extraction ---
vgg = VGGFeatures().to(device)
vgg = vgg.to(device)
print("VGG model initialized")
content_features = vgg(content)
style_features = vgg(style)

# --- Style Gram Matrices ---
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# --- Loss Weights ---
style_weight = 1e6
content_weight = 1e0

# --- Optimizer ---
optimizer = optim.LBFGS([target])

# --- Run Style Transfer ---
epochs = 100

def closure():
    optimizer.zero_grad()
    target_features = vgg(target)

    # Content loss
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

    # Style loss
    style_loss = 0
    for layer in style_grams:
        target_gram = gram_matrix(target_features[layer])
        style_gram = style_grams[layer]
        style_loss += torch.mean((target_gram - style_gram)**2)

    total_loss = content_weight * content_loss + style_weight * style_loss
    total_loss.backward()
    return total_loss

for i in range(epochs):
    optimizer.step(closure)
    if i % 50 == 0:
        print(f"Iteration {i}")

# --- Show Image ---
final_img = target.clone().detach().cpu().squeeze()
final_img = final_img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
final_img = final_img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
final_img = final_img.clamp(0, 1)
plt.imshow(final_img.permute(1, 2, 0))
plt.title("Stylized Image")
plt.axis('off')
plt.show(block=True)

# --- Save Image ---
plt.imsave("output_image.jpg", final_img.permute(1, 2, 0).numpy())
print("Image saved as output_image.jpg")
