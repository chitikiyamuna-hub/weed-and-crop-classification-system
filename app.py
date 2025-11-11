import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import gradio as gr
import os # <-- NEW: Import the OS module

# 1. Define the model class (must be the same as in model.py)
class WeedCropCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Set pretrained=False when loading weights, as the weights file
        # will contain the full state (including the final FC layer).
        self.base_model = models.resnet18(pretrained=False) 
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# 2. Setup transformation and load model
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Instantiate model and load weights
MODEL_PATH = "weed_crop_model.pth"
class_names = ['Crop', 'Weed'] 
NUM_CLASSES = len(class_names)

model = WeedCropCNN(num_classes=NUM_CLASSES)
# Use CPU if GPU is not available on the deployment server
device = torch.device("cpu") 
model.to(device)

try:
    # Load weights onto CPU to ensure compatibility with most servers
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print("WARNING: Model weights (weed_crop_model.pth) not found. Cannot run inference.")
    
# 3. Inference function
def classify_image(image):
    if image is None:
        # Check if the model weights were loaded before proceeding
        if 'eval' not in model.__dict__.keys(): 
             return {"Error": "Model weights not loaded."}
        return "Please upload an image."

    # Preprocess the image
    img_t = preprocess(image)
    batch_t = torch.unsqueeze(img_t, 0).to(device) # Add batch dimension and move to device

    # Run inference
    with torch.no_grad():
        out = model(batch_t)
        probabilities = torch.nn.functional.softmax(out[0], dim=0)

    # Get prediction and format output
    confidences = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    return confidences

# 4. Gradio Interface
# --- CRITICAL DEPLOYMENT CHANGE BELOW ---
# Read the port from the environment variable (e.g., set by Render)
# Use the Gradio environment variable names for clarity, though os.environ.get('PORT') 
# is also common if not using GRADIO_SERVER_PORT.
server_port = int(os.environ.get('GRADIO_SERVER_PORT', 7860)) 
server_name = os.environ.get('GRADIO_SERVER_NAME', "0.0.0.0")

gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload Crop/Weed Image"),
    outputs=gr.Label(num_top_classes=NUM_CLASSES),
    title="Weed and Crop Classifier (ResNet18)",
    description="Upload an image to classify it as 'Crop' or 'Weed'.",
    allow_flagging="never"
).launch(
    server_name=server_name,
    server_port=server_port 
)
