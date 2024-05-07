import sys
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torchvision import models
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

stats = ((0.5832, 0.5832, 0.5832), (0.1413, 0.1413, 0.1413))
# Mean:  tensor([0.5832, 0.5832, 0.5832])
# Std:  tensor([0.1413, 0.1413, 0.1413])
transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=stats[0], std=stats[1]),
])


class XRayClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(XRayClassifier, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        # Where we define all the parts of the model
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        resnet_out_size = self.base_model.fc.in_features
        # Make a classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(resnet_out_size, num_classes)
        )

    def forward(self, x):
        # Connect these parts and return the output
        x = self.features(x)
        output = self.classifier(x)
        return output


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


device = get_default_device()


def calculate_accuracy(output_vector, true_label):
    predicted_label = np.argmax(output_vector)
    return 1 if predicted_label == true_label else 0


# Load and preprocess the image
def preprocess_image(image_path, transform):
    print(image_path)
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)


# Predict using the model
def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()


import matplotlib.pyplot as plt


def visualize_predictions(original_image, probabilities, class_names):
    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))

    # Display image
    print(f"vizualize_predictions={original_image}")
    axarr[0].imshow(original_image)
    print(f"1")
    axarr[0].axis("off")


    # Display predictions
    axarr[1].barh(class_names, probabilities)
    print(f"2")
    axarr[1].set_xlabel("Probability")
    print(f"3")
    axarr[1].set_title("Class Predictions")
    print(f"4")
    axarr[1].set_xlim(0, 1)

    # plt.tight_layout()
    # plt.show()
    # Return the matplotlib figure
    print("return")
    return fig


model = XRayClassifier(num_classes=2)
model.load_state_dict(torch.load("D:/XRay/chest_x_ray_resnet18_best_version.pth", map_location=device))

def classify_image(image_path):
    original_image, image_tensor = preprocess_image(image_path, transform)
    probabilities = predict(model, image_tensor, device)

    return probabilities

