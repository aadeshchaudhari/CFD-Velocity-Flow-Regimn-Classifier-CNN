import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from model.model import VelocityFlowRegimnCNNWithRegularization

trained_model = None
class_names = [
    'High Speed Aerofoil',
    'High speed Elliptic',
    'Low Speed Aerofoil',
    'Low Speed Elliptic'
]

# Load the pre-trained CNN model
def load_model():
    global trained_model

    if trained_model is None:
        trained_model = VelocityFlowRegimnCNNWithRegularization(num_classes=len(class_names))
        trained_model.load_state_dict(torch.load("saved_model.pth", map_location="cpu"))
        trained_model.eval()

    return trained_model


# ---------- Prediction Function ----------
def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    # --- FIXED TRANSFORMS FOR INFERENCE ---
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # ONLY resize
        transforms.ToTensor(),          # convert to tensor
        transforms.Normalize(           # same normalization as training
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Apply transforms correctly
    image_tensor = image_transforms(image).unsqueeze(0)  # <<< FIXED

    model = load_model()
    model.eval()  # ensure evaluation mode

    with torch.no_grad():
        outputs = model(image_tensor)
        _, pred = torch.max(outputs, 1)
        return class_names[pred.item()]
