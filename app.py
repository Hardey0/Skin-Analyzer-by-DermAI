import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 7)
model.load_state_dict(torch.load("model/best_model.pth", map_location=device))
model.to(device)
model.eval()

# Class names
class_names = [
    'Actinic Keratosis',
    'Basal Cell Carcinoma',
    'Benign Keratosis',
    'Dermatofibroma',
    'Melanocytic Nevi',
    'Melanoma',
    'Vascular Lesions'
]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Prediction function
def predict(image):
    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred_index = output.argmax().item()
        prediction = class_names[pred_index]
    return f"Prediction: {prediction}"

# Gradio Interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="DermAI - Skin Disease Detection",
    description="Upload an image of a skin lesion to detect the condition."
)

if __name__ == "__main__":
   iface.launch(enable_queue=False)
