# Description:
# This script is used to initialize different types of DenseNet models based on the given name.

# =================================================================================================

import torch
from torch import nn
from torchvision import models
from torchvision.models import (
    DenseNet121_Weights, 
    DenseNet161_Weights, 
    DenseNet169_Weights, 
    DenseNet201_Weights,  
    DenseNet)
from typing import Tuple

# =================================================================================================

def get_pt_model(model_name: str, out_dim: int, device: torch.device) -> Tuple[DenseNet, str]:
    """Initialize different types of DenseNet models based on given name.

    Args:
        model_name (str): Name of the DenseNet structure
        out_dim (int): Output dimensions
        device (torch.device): Device to run the model on

    Raises:
        ValueError: If model_name is not recognized
        
    Returns:
        Tuple[DenseNet, str]: DenseNet Model and the model name

    Available Models:
        - DenseNet121
        - DenseNet161
        - DenseNet169
        - DenseNet201
    """


    if model_name == 'DenseNet_121':
        model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)

    elif model_name == 'DenseNet_161':
        model = models.densenet161(weights=DenseNet161_Weights.DEFAULT)

    elif model_name == 'DenseNet_169':
        model = models.densenet169(weights=DenseNet169_Weights.DEFAULT)

    elif model_name == 'DenseNet_201':
        model = models.densenet201(weights=DenseNet201_Weights.DEFAULT)

    else :
        raise ValueError(f"Unknown model name: {model_name}")

    # Modify the fully connected layer to match the number of classes
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, out_dim)

    # Send the model to the device
    model = model.to(device)

    return model, model_name