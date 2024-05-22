import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split
from torch import nn
from tqdm import tqdm

# Create an instance of the ResNet50 model
model = models.resnet50()
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1),  
)

# Load the model state dict from a file
state_dict = torch.load('Performances\ResNet KIP\Model-Files\ResNet50 - Tv2 - 1.000000e-04.pt', map_location=torch.device('cpu'))

# Load the state dict into the model
model.load_state_dict(state_dict)

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),

    # Transform into 3 channel pseudo RGB
    transforms.Grayscale(num_output_channels=3),

    transforms.ToTensor(),

    # Normalization parameters for ImageNet
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(root='Data/KIP Extended', transform=transform)

# Calculate the percentage of the dataset to sample
sample_percentage = 0.5

# Calculate the number of samples to include
num_samples = int(len(test_dataset) * sample_percentage)

# Randomly sample the dataset
test_dataset = random_split(test_dataset, [num_samples, len(test_dataset) - num_samples])[0]

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



model.eval()
test_accuracy = 0
with torch.no_grad():
    for inputs, labels in tqdm(test_loader):
        outputs = model(inputs).view(-1)

        # Calculate accuracy
        preds = torch.round(torch.sigmoid(outputs))
        test_accuracy += torch.sum(preds == labels.data)

test_accuracy = float(test_accuracy / len(test_dataset))

print(f"Test accuracy: {test_accuracy:.4f}")
