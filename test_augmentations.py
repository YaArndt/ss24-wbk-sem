# Description:
# This script is used to visualize the data augmentation transformations on the dataset.
# It can be used to check if the transformations are working as expected.

# =================================================================================================

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from utils.data	import DataVisualizer, TransformingDataset
from torchvision import datasets
import config

# =================================================================================================

# Some constants
DATA_DIR = '01_data_selfmade\kurz'
RUN_DIR = '01_runs'
RUN_TAG = 'DataAugmentation'
AUGMENTATION_EPOCHS = 1

# Define the transformations for the training dataset
train_augmentation = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.RandomRotation((-360, 360)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
    transforms.ColorJitter(brightness=0.5),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),

    # # Normalization parameters for ImageNet
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the transformations for the testing dataset
test_augmentation = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.RandomRotation((-360, 360)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
    transforms.ColorJitter(brightness=0.5),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),

    # # Normalization parameters for ImageNet
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
data = datasets.ImageFolder(root=DATA_DIR)

# Split dataset into training and testing
train_size = int(config.TRAIN_SIZE_RATIO * len(data))
test_size = len(data) - train_size
train_data, test_data = random_split(data, [train_size, test_size])

# Apply the transformations to the datasets using Subset and lambda function
train_data = TransformingDataset(train_data, train_augmentation)

test_data = TransformingDataset(test_data, test_augmentation)

# Create DataLoader for training and testing datasets
train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE)
test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE)

# Initialize the writer for the data samples
data_writer = SummaryWriter(f"{RUN_DIR}/{RUN_TAG}/Data")

# Initialize the data visualizer
# The class dictionary is inverted to map the class indices to class names
viz = DataVisualizer(class_dict={y: x for x, y in data.class_to_idx.items()})

train_i = 0
test_i = 0

for _ in range(AUGMENTATION_EPOCHS):

    for images, labels in train_loader:
        train_i += 1
        log_image = viz.log_image(images, labels)
        data_writer.add_image('Train Images', log_image, train_i)
        

    for images, labels in test_loader:
        test_i += 1
        log_image = viz.log_image(images, labels)
        data_writer.add_image('Test Images', log_image, test_i)
        