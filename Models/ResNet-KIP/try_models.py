# Description: 
# Try out different model architectures in a very basic setup
# to determine the best candidates for further optimizattion.

# =================================================================================================

import torch
import torchvision.transforms as transforms

from torchvision.models import ResNet18_Weights
from torchvision.models import ResNet34_Weights
from torchvision.models import ResNet50_Weights
from torchvision.models import ResNet101_Weights
from torchvision.models import ResNet152_Weights
from torchvision.models import ResNet
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision

from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from PIL import Image, ImageOps
import pandas as pd

# =================================================================================================

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),

    # Transform into 3 channel pseudo RGB
    transforms.Grayscale(num_output_channels=3),

    transforms.ToTensor(),

    # Normalization parameters for ImageNet
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load your dataset
dataset = datasets.ImageFolder(root='Data/KIP', transform=transform)

# Split dataset into training and testing
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

print(f"Train size: {len(train_dataset)}")
print(f"Test size: {len(test_dataset)}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# If using GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def initialize_model(model_name: str, out_dim: int) -> ResNet:
    """Initialize different types of ResNet models based on given name.

    Args:
        model_name (str): Name of the ResNet structure
        out_dim (int): Output dimensions

    Returns:
        ResNet: ResNet Model
    """


    if model_name == 'ResNet18':
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    elif model_name == 'ResNet34':
        model = models.resnet34(weights=ResNet34_Weights.DEFAULT)

    elif model_name == 'ResNet50':
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    elif model_name == 'ResNet101':
        model = models.resnet101(weights=ResNet101_Weights.DEFAULT)

    elif model_name == 'ResNet152':
        model = models.resnet152(weights=ResNet152_Weights.DEFAULT)

    # Modify the fully connected layer to match the number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, out_dim),  
    )

    model = model.to(device)
    return model

def initialize_optimizer(model: torchvision.models.ResNet, learning_rate=0.001) -> optim.Adam:
    """Initialize basic Adam optimizer.

    Args:
        model (torchvision.models.ResNet): Model to optimize
        learning_rate (float, optional): Learning rate. Defaults to 0.001.

    Returns:
        optim.Adam: Optimizer object.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return optimizer

def initialize_loss_function() -> nn.BCEWithLogitsLoss:
    """Initialize basic loss function.

    Returns:
        nn.BCEWithLogitsLoss: Loss function object.
    """
    return nn.BCEWithLogitsLoss()

def initialize_scheduler(optimizer, epochs_to_run):
    return CosineAnnealingLR(optimizer, T_max=epochs_to_run, eta_min=0)

# Define model tryout grid
# models_to_train = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]
models_to_train = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]
lrs_to_try = [0.001, 0.0001]
epochs_to_run = 30

# Relevant performance metrics to be used in the creation of the performance DataFrames
performance_frame_columns = ["Epoch", "Train Loss", "Test Loss", "Test Accuracy"]

# Goes through the tryout grid
for model_name in models_to_train:
    for lr in lrs_to_try:
        
        # Holds the entries per epoch
        performance_frame_data = []
        
        print(f"Training {model_name} with lr {format(lr, 'e')}")

        # Get relevant objects
        model = initialize_model(model_name, 1)
        optimizer = initialize_optimizer(model)
        criterion = initialize_loss_function()
        scheduler = initialize_scheduler(optimizer, epochs_to_run)

        # Goes through the epochs
        for epoch in range(epochs_to_run):
            
            # Training phase
            model.train() 
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = model(inputs).view(-1)
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

            scheduler.step()

            # Validation phase
            model.eval()  # Set model to evaluate mode
            test_loss = 0.0
            test_accuracy = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs).view(-1)
                    loss = criterion(outputs, labels.float())
                    test_loss += loss.item()

                    # Calculate accuracy
                    preds = torch.round(torch.sigmoid(outputs))
                    test_accuracy += torch.sum(preds == labels.data)

            # Calculate performance statistics
            loss = float(running_loss / len(train_loader))
            test_loss = float(test_loss / len(test_loader))
            test_accuracy = float(test_accuracy / len(test_dataset))

            # Log statistics
            performance_frame_data.append([epoch + 1, loss, test_loss, test_accuracy])

            # Print statistics
            print(f'{model_name} | Epoch {epoch+1}, Train Loss: {round(loss, 6)}, Test Loss: {round(test_loss, 6)}, Test Acc: {round(test_accuracy, 6)}')

        # Create statistics frame
        performance_frame = pd.DataFrame(data=performance_frame_data, columns=performance_frame_columns)
        file_name = f"Performances/ResNet KIP/CSVs/Model Performance (Transforms V2) - E{epochs_to_run} - {model_name} - {format(lr, 'e')}.csv"
        performance_frame.to_csv(file_name, sep=";", index=False)
            
        torch.save(model.state_dict(), f"Performances/ResNet KIP/Model-Files/{model_name} - Tv2 - {format(lr, 'e')}.pt")

        print("DONE!")

        # Clear CUDA cache to prevent bleeding
        torch.cuda.empty_cache()

print("FINISHED!")