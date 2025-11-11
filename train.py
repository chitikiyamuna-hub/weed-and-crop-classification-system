import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import WeedCropCNN # Assumes model.py is in the same directory

# Transform images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    # NOTE: It's highly recommended to add normalization here for pre-trained models:
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
# Ensure your dataset is structured like: dataset/crop/image.jpg and dataset/weed/image.jpg
train_dataset = datasets.ImageFolder("dataset/", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Initialize model
# Automatically sets num_classes based on the number of folders in "dataset/"
num_classes = len(train_dataset.classes)
model = WeedCropCNN(num_classes=num_classes)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
print(f"Starting training on device: {device}")
for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # Move data to the same device as the model
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
