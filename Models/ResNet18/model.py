import torch
import torchvision.transforms as transforms
import torchvision.utils
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

def make_square(image):
    # Calculate padding to make the image square
    max_dim = max(image.size)
    left = (max_dim - image.width) // 2
    top = (max_dim - image.height) // 2
    right = max_dim - image.width - left
    bottom = max_dim - image.height - top
    
    # Pad the image and return
    return ImageOps.expand(image, (left, top, right, bottom), fill=0)  # You can change the fill color if needed


transform = transforms.Compose([
    transforms.Lambda(make_square),
    transforms.Grayscale(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to pseudo-RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization parameters for ImageNet
])

# Load your dataset
dataset = datasets.ImageFolder(root='Data\BilderNeu', transform=transform)

# Split dataset into training and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

classes = dataset.classes

def show_image(i: int):
    elem = dataset.__getitem__(i)
    Image.Image.show(transforms.ToPILImage()(elem[0]))

# Load a pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)

# Modify the fully connected layer to the number of classes you have (e.g., binary classification)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2) 

# If using GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


criterion = nn.BCEWithLogitsLoss()  # Use CrossEntropyLoss for multi-class
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 10  # Set the number of epochs

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels.float())  # labels.float() is for BCEWithLogitsLoss
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    # Validation phase
    model.eval()  # Set model to evaluate mode
    val_loss = 0.0
    val_accuracy = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            val_loss += loss.item()
            # Calculate accuracy
            preds = torch.round(torch.sigmoid(outputs))
            val_accuracy += torch.sum(preds == labels.data)

    # Print statistics
    print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}, Validation Accuracy: {val_accuracy / len(val_dataset)}')
