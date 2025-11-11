import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import gradio as gr
import os # <-- Necessary for reading environment variables

# --- 1. Model Definition ---
# Must exactly match the class definition in model.py
class WeedCropCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Set pretrained=False when loading local weights
        self.base_model = models.resnet18(pretrained=False) 
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# --- 2. Setup and Loading ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

MODEL_PATH = "weed_crop_model.pth"
class_names = ['Crop', 'Weed'] 
NUM_CLASSES = len(class_names)

model = WeedCropCNN(num_classes=NUM_CLASSES)
# Force CPU usage on deployment server to avoid GPU dependency issues
device = torch.device("cpu") 
model.to(device)

try:
    # Load weights onto the defined device
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print("WARNING: Model weights (weed_crop_model.pth) not found. Cannot run inference.")
    
# --- 3. Inference Function ---
def classify_image(image):
    if image is None:
        return "Please upload an image."

    # Initial check if model is ready
    if not model.training and 'eval' not in model.__dict__.keys():
         return {"Error": "Model weights failed to load."}

    # Preprocess the image
    img_t = preprocess(image)
    batch_t = torch.unsqueeze(img_t, 0).to(device) 

    # Run inference
    with torch.no_grad():
        out = model(batch_t)
        probabilities = torch.nn.functional.softmax(out[0], dim=0)

    confidences = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    return confidences

# --- 4. Gradio Interface and Deployment Launch ---

# CRITICAL FIX: Read the host and port from environment variables provided by the host.
# Render primarily uses the 'PORT' variable, but we check Gradio's variables as a fallback.
# The host must be set to 0.0.0.0 for external access.
server_port = int(os.environ.get('PORT', os.environ.get('GRADIO_SERVER_PORT', 7860))) 
server_name = os.environ.get('GRADIO_SERVER_NAME', "0.0.0.0")

print(f"Attempting to launch server on {server_name}:{server_port}")

if __name__ == "__main__":
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
