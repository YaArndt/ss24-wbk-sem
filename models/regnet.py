# Description:
# This script is used to initialize different types of RegNet models based on the given name.

# =================================================================================================

import torch
from torch import nn
from torchvision import models
from torchvision.models import (
    RegNet_X_400MF_Weights, 
    RegNet_X_800MF_Weights, 
    RegNet_X_1_6GF_Weights, 
    RegNet_X_3_2GF_Weights, 
    RegNet_X_8GF_Weights,
    RegNet_X_16GF_Weights,
    RegNet_X_32GF_Weights,
    RegNet_Y_400MF_Weights, 
    RegNet_Y_800MF_Weights, 
    RegNet_Y_1_6GF_Weights, 
    RegNet_Y_3_2GF_Weights, 
    RegNet_Y_8GF_Weights,
    RegNet_Y_16GF_Weights,
    RegNet_Y_32GF_Weights,
    RegNet_Y_128GF_Weights,  
    RegNet)
from typing import Tuple

# =================================================================================================

def get_pt_model(model_name: str, out_dim: int, device: torch.device) -> Tuple[RegNet, str]:
    """Initialize different types of RegNet models based on given name.

    Args:
        model_name (str): Name of the RegNet structure
        out_dim (int): Output dimensions
        device (torch.device): Device to run the model on

    Raises:
        ValueError: If model_name is not recognized
        
    Returns:
        Tuple[RegNet, str]: RegNet Model and the model name

    Available Models:
        RegNet_X_400MF, 
        RegNet_X_800MF, 
        RegNet_X_1_6GF, 
        RegNet_X_3_2GF, 
        RegNet_X_8GF,
        RegNet_X_16GF,
        RegNet_X_32GF,
        RegNet_Y_400MF, 
        RegNet_Y_800MF, 
        RegNet_Y_1_6GF, 
        RegNet_Y_3_2GF, 
        RegNet_Y_8GF,
        RegNet_Y_16GF,
        RegNet_Y_32GF,
        RegNet_Y_128GF,  
    """


    if model_name == 'RegNet_X_400MF':
        model = models.resnet18(weights=RegNet_X_400MF_Weights.DEFAULT)

    elif model_name == 'RegNet_X_800MF':
        model = models.resnet34(weights=RegNet_X_800MF_Weights.DEFAULT)

    elif model_name == 'RegNet_X_1_6GF':
        model = models.resnet50(weights=RegNet_X_1_6GF_Weights.DEFAULT)

    elif model_name == 'RegNet_X_3_2GF':
        model = models.resnet101(weights=RegNet_X_3_2GF_Weights.DEFAULT)

    elif model_name == 'RegNet_X_8GF':
        model = models.resnet152(weights=RegNet_X_8GF_Weights.DEFAULT)

    elif model_name == 'RegNet_X_16GF':
        model = models.resnet152(weights=RegNet_X_16GF_Weights.DEFAULT)
    
    elif model_name == 'RegNet_X_32GF':
        model = models.resnet152(weights=RegNet_X_32GF_Weights.DEFAULT)
    
    elif model_name == 'RegNet_Y_400MF':
        model = models.resnet18(weights=RegNet_Y_400MF_Weights.DEFAULT)

    elif model_name == 'RegNet_Y_800MF':
        model = models.resnet34(weights=RegNet_Y_800MF_Weights.DEFAULT)

    elif model_name == 'RegNet_Y_1_6GF':
        model = models.resnet50(weights=RegNet_Y_1_6GF_Weights.DEFAULT)

    elif model_name == 'RegNet_Y_3_2GF':
        model = models.resnet101(weights=RegNet_Y_3_2GF_Weights.DEFAULT)

    elif model_name == 'RegNet_Y_8GF':
        model = models.resnet152(weights=RegNet_Y_8GF_Weights.DEFAULT)

    elif model_name == 'RegNet_Y_16GF':
        model = models.resnet152(weights=RegNet_Y_16GF_Weights.DEFAULT)
    
    elif model_name == 'RegNet_Y_32GF':
        model = models.resnet152(weights=RegNet_Y_32GF_Weights.DEFAULT)
    
    elif model_name == 'RegNet_Y_128GF':
        model = models.resnet152(weights=RegNet_Y_128GF_Weights.DEFAULT)
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