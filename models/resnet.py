# Description:
# This script is used to initialize different types of ResNet models based on the given name.

# =================================================================================================

import torch
from torch import nn
from torchvision import models
from torchvision.models import (
    ResNet18_Weights, 
    ResNet34_Weights, 
    ResNet50_Weights, 
    ResNet101_Weights, 
    ResNet152_Weights, 
    ResNet)
from typing import Tuple

# =================================================================================================

def get_pt_model(model_name: str, out_dim: int, device: torch.device) -> Tuple[ResNet, str]:
    """Initialize different types of ResNet models based on given name.

    Args:
        model_name (str): Name of the ResNet structure
        out_dim (int): Output dimensions
        device (torch.device): Device to run the model on

    Raises:
        ValueError: If model_name is not recognized
        
    Returns:
        Tuple[ResNet, str]: ResNet Model and the model name

    Available Models:
        - ResNet18
        - ResNet34
        - ResNet50
        - ResNet101
        - ResNet152
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
    else :
        raise ValueError(f"Unknown model name: {model_name}")

    # Modify the fully connected layer to match the number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, out_dim),  
    )

    # Send the model to the device
    model = model.to(device)

    return model, model_name