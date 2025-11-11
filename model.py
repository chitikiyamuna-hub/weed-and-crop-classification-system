import torch.nn as nn
import torchvision.models as models

class WeedCropCNN(nn.Module):
    def __init__(self, num_classes=2): # FIX: Corrected to __init__
        super().__init__()
        # Use a pre-trained ResNet-18 model as the base
        self.base_model = models.resnet18(pretrained=True)
        # Replace the final fully connected layer (fc) for the new number of classes
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
