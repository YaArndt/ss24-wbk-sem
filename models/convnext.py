# Description:
# This script is used to initialize different types of ConvNeXt models based on the given name.

# =================================================================================================

import torch
from torch import nn
from torchvision import models
from torchvision.models import (
    ConvNeXt_Tiny_Weights, 
    ConvNeXt_Small_Weights, 
    ConvNeXt_Base_Weights, 
    ConvNeXt_Large_Weights,  
    ConvNeXt)
from typing import Tuple

# =================================================================================================

def get_pt_model(model_name: str, out_dim: int, device: torch.device) -> Tuple[ConvNeXt, str]:
    """Initialize different types of ConvNeXt models based on given name.

    Args:
        model_name (str): Name of the ConvNeXt structure
        out_dim (int): Output dimensions
        device (torch.device): Device to run the model on

    Raises:
        ValueError: If model_name is not recognized
        
    Returns:
        Tuple[ConvNeXt, str]: ConvNeXt Model and the model name

    Available Models:
        - ConvNeXt_Tiny
        - ConvNeXt_Small
        - ConvNeXt_Base
        - ConvNeXt_Large
    """


    if model_name == 'ConvNeXt_Tiny':
        model = models.resnet18(weights=ConvNeXt_Tiny_Weights.DEFAULT)

    elif model_name == 'ConvNeXt_Small':
        model = models.resnet34(weights=ConvNeXt_Small_Weights.DEFAULT)

    elif model_name == 'ConvNeXt_Base':
        model = models.resnet50(weights=ConvNeXt_Base_Weights.DEFAULT)

    elif model_name == 'ConvNeXt_Large':
        model = models.resnet101(weights=ConvNeXt_Large_Weights.DEFAULT)

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