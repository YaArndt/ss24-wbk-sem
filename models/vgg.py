# Description:
# This script is used to initialize different types of VGG models based on the given name.

# =================================================================================================

import torch
from torch import nn
from torchvision import models
from torchvision.models import (
    VGG11_Weights,
    VGG11_BN_Weights,
    VGG13_Weights,
    VGG13_BN_Weights,
    VGG16_Weights,
    VGG16_BN_Weights,
    VGG19_Weights,
    VGG19_BN_Weights,
    VGG)
from typing import Tuple

# =================================================================================================

def get_pt_model(model_name: str, out_dim: int, device: torch.device) -> Tuple[VGG, str]:
    """Initialize different types of VGG models based on given name.    

    Args:
        model_name (str): Name of the VGG structure
        out_dim (int): Output dimensions
        device (torch.device): Device to run the model on

    Raises:
        ValueError: If the model_name is not recognized 

    Returns:
        Tuple[VGG, str]: The model and the model name

    Available Models:
        - VGG11
        - VGG11-BN
        - VGG13
        - VGG13-BN
        - VGG16
        - VGG16-BN
        - VGG19
        - VGG19-BN

    Models with BN have Batch Normalization layers.
    """

    if model_name == 'VGG11':
        model = models.vgg11(weights=VGG11_Weights.DEFAULT)

    elif model_name == 'VGG11-BN':
        model = models.vgg11_bn(weights=VGG11_BN_Weights.DEFAULT)

    elif model_name == 'VGG13':
        model = models.vgg13(weights=VGG13_Weights.DEFAULT)

    elif model_name == 'VGG13-BN':
        model = models.vgg13_bn(weights=VGG13_BN_Weights.DEFAULT)

    elif model_name == 'VGG16':
        model = models.vgg16(weights=VGG16_Weights.DEFAULT)

    elif model_name == 'VGG16-BN':
        model = models.vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)

    elif model_name == 'VGG19':
        model = models.vgg19(weights=VGG19_Weights.DEFAULT)

    elif model_name == 'VGG19-BN':
        model = models.vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)

    else :
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Modify the fully connected layer to match the number of classes
    model.classifier[-1] = nn.Linear(in_features=4096, out_features=out_dim)

    # Send the model to the device
    model = model.to(device)

    return model, model_name