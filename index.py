import os
import random
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 7)  # 7 skin classes
model.load_state_dict(torch.load("model/best_model.pth", map_location=device))
model.to(device)
model.eval()

# CAM extractor (initialized once)
cam_extractor = SmoothGradCAMpp(model)

# Skin condition class names
class_names = [
    'Actinic Keratosis',
    'Basal Cell Carcinoma',
    'Benign Keratosis',
    'Dermatofibroma',
    'Melanocytic Nevi',
    'Melanoma',
    'Vascular Lesions'
]

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or request.files['image'].filename == '':
        return "No image uploaded", 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Load and transform image
    image = Image.open(filepath).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    input_tensor.requires_grad = True  # Enable gradients for CAM

    # Forward pass (no torch.no_grad!)
    output = model(input_tensor)
    pred_idx = output.argmax(dim=1).item()
    prediction = class_names[pred_idx]

    # Generate CAM
    cam_map = cam_extractor(pred_idx, output)[0]  # Get CAM for predicted class
    cam_map = cam_map.cpu()
    heatmap = to_pil_image(cam_map, mode='F').resize(image.size)

    # Overlay CAM on original image
    result_image = overlay_mask(image, heatmap, alpha=0.5)

    # Save blended CAM image
    cam_filename = f"cam_{filename}"
    cam_path = os.path.join(app.config['UPLOAD_FOLDER'], cam_filename)
    result_image.save(cam_path)

    # Get two random alternative predictions
    alternatives = [cls for cls in class_names if cls != prediction]
    alt_1, alt_2 = random.sample(alternatives, 2)

    return render_template(
        'result.html',
        prediction=prediction,
        alt_1=alt_1,
        alt_2=alt_2,
        image_url=filepath,
        heatmap_url=cam_path
    )

if __name__ == "__main__":
    app.run(debug=True)
    