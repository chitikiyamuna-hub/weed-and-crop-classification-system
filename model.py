import torch.nn as nn
import torchvision.models as models

class WeedCropCNN(nn.Module):
    # FIX: Corrected __init__
    def __init__(self, num_classes=2): 
        super().__init__()
        # Load pre-trained ResNet-18
        self.base_model = models.resnet18(pretrained=True)
        # Replace the final fully connected layer for custom number of classes
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
