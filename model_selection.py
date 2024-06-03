# Description: 
# This script is used to train and evaluate the models on the dataset.
# It can be used to try out different hyperparameters and models and logs the results to TensorBoard.

# =================================================================================================

import config
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
import torchvision.transforms as transforms
from utils.parameters import ParameterGrid
from utils.performance import Performance
from utils.data	import TransformingDataset, DataVisualizer
from models.resnet import get_pt_model
from models.training import train_model

# =================================================================================================

DATA_DIR = '01_data_selfmade/kurz'
RUN_DIR = '01_runs'
RUN_TAG = 'ResNet_With_Offset_Augmentation'
SAVE_MODELS = True

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Normalization parameters for ImageNet
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

    # Normalization parameters for ImageNet
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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


# Initialize the performance object
pos_label = data.class_to_idx[config.CLASSES["POS"]]
neg_label = data.class_to_idx[config.CLASSES["NEG"]]
performance = Performance(pos_label, neg_label)

# Create a grid of hyperparameters and models
grid = ParameterGrid(
    model = [
        # get_pt_model('ResNet18', 1, device),
        get_pt_model('ResNet34', 1, device)
        # get_pt_model('ResNet50', 1, device),
        # get_pt_model('ResNet101', 1, device),
        # get_pt_model('ResNet152', 1, device)
    ],

    lr = [
        0.0005
        # 0.0001,
    ]
)

# Calculate the number of models to train and keep track of the current model
models_to_train = len(grid)
current_model = 0

# Initialize the writer for the data samples
data_writer = SummaryWriter(f"{RUN_DIR}/{RUN_TAG}/Data")

# Initialize the data visualizer
# The class dictionary is inverted to map the class indices to class names
viz = DataVisualizer(class_dict={y: x for x, y in data.class_to_idx.items()})

# Log a sample of the training and testing images
for images, labels in train_loader:
    log_image = viz.log_image(images, labels)
    data_writer.add_image('Train Images', log_image, 0)
    break

for images, labels in test_loader:
    log_image = viz.log_image(images, labels)
    data_writer.add_image('Test Images', log_image, 0)
    break

# Close the data writer
data_writer.close()

# Iterate over the grid of hyperparameters
for params in grid:
    current_model += 1

    # Get the model and model tag from the parameters
    model: nn.Module = params['model'][0]
    model_tag = params['model'][1]

    # Set the model logging directory
    save_dir = f"{RUN_DIR}/{RUN_TAG}/{model_tag}/{params['lr']}/"
    writer = SummaryWriter(save_dir)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    # Initialize the scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    # Provide some information about the current model
    print(f"Training model {current_model}/{models_to_train}:")
    print(f"Model: {model_tag}, LR: {params['lr']}")

    try:

        # Train the model
        train_model(
            train_loader, 
            test_loader, 
            model, 
            optimizer, 
            writer, 
            performance, 
            scheduler
        )       
        
        # Save the model state dictionary
        if SAVE_MODELS:
            torch.save(model.state_dict(), f"{save_dir}/model_state_dict.pth")

    except KeyboardInterrupt:
        print("Training interrupted by user")
        
        save = ""
        while save not in {'y', 'n', 'yes', 'no'}:
            save = input("Do you want to save the model? (y/n): ").lower()

        if save in {'y', 'yes'}:
            torch.save(model.state_dict(), f"{save_dir}/model_state_dict.pth")
            print("Model saved")

        if current_model < models_to_train:
            continue_training = ""
            while continue_training not in {'y', 'n', 'yes', 'no'}:
                continue_training = input("Do you want to continue training other mdoels? (y/n): ").lower()

            if continue_training in {'n', 'no'}:
                break
    
    except Exception:
        if SAVE_MODELS:
            torch.save(model.state_dict(), f"{save_dir}/model_state_dict.pth")
        


