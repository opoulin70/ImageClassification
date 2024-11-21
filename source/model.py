import torch
import torch.nn as nn
import torchvision.models as models


# Function to load the pretrained model (ResNet18) and modify the last layer to match number of classes in CIFAR-10 (10)
def load_model(num_classes=10):
    # Load the pretrained ResNet18 model
    model = models.resnet18(pretrained=True)

    # Modify the final fully connected layer to match the number of classes in CIFAR-10
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Move the model to the GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    return model
