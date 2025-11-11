import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import WeedCropCNN # Import the model class

# Define transformations for training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Recommended normalization for models pre-trained on ImageNet:
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset: assumes images are structured as 'dataset/class_name/image.jpg'
try:
    train_dataset = datasets.ImageFolder("dataset/", transform=transform)
except FileNotFoundError:
    print("Error: 'dataset/' folder not found. Please create it with 'crop/' and 'weed/' subfolders.")
    exit()
    
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Initialize model
num_classes = len(train_dataset.classes)
model = WeedCropCNN(num_classes=num_classes)

# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
print(f"Starting training on device: {device} for {epochs} epochs...")

for epoch in range(epochs):
    model.train() # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Loss: {running_loss/len(train_loader):.4f}")
    print(f"Accuracy: {100 * correct/total:.2f}%\n")

# Save model weights
torch.save(model.state_dict(), "weed_crop_model.pth")
print("Model saved as weed_crop_model.pth")
