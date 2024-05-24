# Description:
# This script is used to train and evaluate the models on the dataset.

# =================================================================================================

import config
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Optional
from utils.performance import Performance

# =================================================================================================

def train_model (
    train_loader: DataLoader,
    test_loader: DataLoader,
    model: nn.Module,
    optimizer: Optimizer,
    writer: SummaryWriter,
    performance: Performance,
    scheduler: Optional[_LRScheduler] = None,

):
    """Train the model on the given dataset and evaluate on the test dataset.

    Args:
        train_loader (DataLoader): Training dataset
        test_loader (DataLoader): Testing dataset
        model (nn.Module): Model to be trained
        optimizer (Optimizer): Optimizer to be used
        writer (SummaryWriter): SummaryWriter to log the training
        performance (Performance): Performance object to evaluate the model
        scheduler (Optional[LRScheduler], optional): Learning rate scheduler to use. Defaults to None.
    """
    
    # Set the device
    device = next(model.parameters()).device
    criterion = nn.BCEWithLogitsLoss()

    # Keep track of the number of batches processed in training and testing
    train_i = 0
    test_i = 0

    # Train the model for the given number of epochs
    for _ in tqdm(range(config.EPOCHS)):

        # Train the model
        model.train()
        model.zero_grad

        for images, labels in train_loader:
            train_i += 1

            # Send the data to the device
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients of the optimizer
            optimizer.zero_grad()

            # Forward pass
            output = model(images).view(-1)
            loss = criterion(output, labels.float())

            # Backward pass
            loss.backward()
            optimizer.step()

            # Log the loss
            writer.add_scalar("Train/Loss", loss, train_i)

        # If needed, update the learning rate
        if scheduler is not None:
            scheduler.step()

        # Test the model
        model.eval()

        for images, labels in test_loader:
            test_i += 1

            # Send the data to the device
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            output = model(images).view(-1)
            loss = criterion(output, labels.float())

            # Calculate the predictions
            preds = torch.round(torch.sigmoid(output)).int()

            acc = performance.accuracy(preds, labels)
            prec = performance.precision(preds, labels)
            rec = performance.recall(preds, labels)
            f1 = performance.f1_score(preds, labels)
            
            # Log the loss and performance metrics
            writer.add_scalar("Test/Loss", loss, test_i)
            writer.add_scalar("Test/Accuracy", acc, test_i)
            writer.add_scalar("Test/Precision", prec, test_i)
            writer.add_scalar("Test/Recall", rec, test_i)
            writer.add_scalar("Test/F1 Score", f1, test_i)