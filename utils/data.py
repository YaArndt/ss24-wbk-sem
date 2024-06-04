# Description:
# Defines a set of data augmentation transformations to be applied to the images in the dataset.

# =================================================================================================

from PIL import Image, ImageOps
import random
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision.transforms import transforms
from torchvision import datasets
import io
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from typing import Tuple, List

# =================================================================================================

class KTimes90Rotation:
    def __call__(self, image: Tensor):
        """Rotates the image tensor by 0, 90, 180, or 270 degrees randomly

        Args:
            image (Tensor): Image tensor to be rotated

        Returns:
            Tensor: Rotated image tensor
        """
        angles = [0, 90, 180, 270]
        angle = random.choice(angles)
        return F.rotate(image, angle)
    
class PadToSquare:
    def __call__(self, image: Image.Image) -> Image.Image:
        """Create a square image from any sized image by padding the shorter sides

        Args:
            image (Image.Image): Original image

        Returns:
            Image.Image: Squared image
        """
        # Calculate padding to make the image square
        max_dim = max(image.size)
        left = (max_dim - image.width) // 2
        top = (max_dim - image.height) // 2
        right = max_dim - image.width - left
        bottom = max_dim - image.height - top
            
        # Pad the image and return
        return ImageOps.expand(image, (left, top, right, bottom), fill=0)

class TransformingDataset(Dataset):
    """Applies a transformation to the images in the dataset
    """
    def __init__(self, dataset, transform=None):
        """Initializes the TransformingDataset object with the dataset and transformation

        Args:
            dataset (dataset): Dataset to be transformed
            transform (transform, optional): Transformations to be applied. Defaults to None.
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
class CombinedBinaryDataset(Dataset):
    """Creates a dataset by combining multiple image folders into a single dataset
    """
    def __init__(self, image_folders: List[str]):
        """Creates a dataset by combining multiple image folders into a single dataset

        Args:
            image_folders (List[str]): List of image folder paths

        Raises:
            ValueError: If the class indices are mismatched
            ValueError: If the number of classes is more than 2
        """
        self.datasets = [datasets.ImageFolder(folder) for folder in image_folders]
        self.cumulative_sizes = torch.cumsum(torch.tensor([len(ds) for ds in self.datasets]), 0)

        self.class_to_idx = {}
        
        # Combine the class_to_idx dictionaries from all datasets
        for i in range(len(self.datasets)):
            for k, v in self.datasets[i].class_to_idx.items():
                if k in self.class_to_idx and self.class_to_idx[k] != v:
                    raise ValueError(f"Key '{k}' has mismatched values!")
                else:
                    self.class_to_idx[k] = v

        if len(self.class_to_idx) > 2:
            raise ValueError("Too many classes in the dataset!")

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of bounds")
        
        dataset_idx = (idx < self.cumulative_sizes).nonzero(as_tuple=True)[0][0].item()
        if dataset_idx > 0:
            idx -= self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][idx]

class DataVisualizer:
    """Handles the visualization of image batches with their corresponding labels
    """

    def __init__(self, class_dict: dict, figsize: Tuple[int, int] = (8, 6)):
        """Initializes the DataVisualizer object with the class dictionary and figure size

        Args:
            class_dict (dict): Dictionary mapping class indices to class names
            figsize (Tuple[int, int], optional): Size of the figures. Defaults to (8, 6).
        """
        self.class_dict = class_dict
        self.figsize = figsize

    def visualize_image_batch(self, batched_images: Tensor, labels: Tensor, grid_size: Tuple[int, int] = (3, 4)) -> Figure:
        """Create a figure with a grid of subplots, each containing an image and its label

        Args:
            batched_images (Tensor): Batch of images to visualize
            labels (Tensor): Labels corresponding to the images
            grid_size (Tuple[int, int], optional): Grid size of the image grid. Defaults to (3, 4).

        Returns:
            Figure: Image grid figure
        """

        max_images = grid_size[0] * grid_size[1]
        used_images = 0

        # Create a figure with a grid of subplots
        fig: Figure
        axes: Axes
        fig, axes = plt.subplots(nrows=grid_size[0], ncols=grid_size[1], figsize=self.figsize, squeeze=False)
        
        # Flatten the axes array for easier iteration (in case of a single row/column)
        axes = axes.flatten()
        
        # Iterate over each image and its label
        for i, (image, label) in enumerate(zip(batched_images, labels)):
            if used_images == max_images:
                break
            
            # Increment the number of used images
            used_images += 1

            # Get the current axis
            ax: Axes = axes[i]

            # Convert the image tensor to a PIL image
            image = transforms.ToPILImage()(image)

            # Plot the image on the corresponding subplot
            ax.imshow(image)
            
            # Set the title of the subplot as the label
            ax.set_title(self.class_dict[label.item()])
            
            # Remove the axis ticks and labels
            ax.axis('off')
        
        # Hide unused subplots if there are any
        for i in range(max_images, len(axes)):
            fig.delaxes(axes[i])

        
        # Adjust the spacing between subplots
        plt.tight_layout()
        
        return fig
    
    def log_image(self, batched_images: Tensor, labels: Tensor, grid_size: Tuple[int, int] = (3, 4)) -> np.ndarray:

        # Create a buffer to store the image
        buffer = io.BytesIO()

        # Visualize the image batch and save the figure to the buffer
        fig = self.visualize_image_batch(batched_images, labels, grid_size)
        fig.savefig(buffer, format='png')
        plt.close(fig)
        buffer.seek(0)

        # Open the image from the buffer
        image = Image.open(buffer)
        
        # Convert the image to a numpy array and transpose the dimensions to fit TensorBoard
        tensorboard_array = np.array(image).transpose(2, 0, 1)
        
        return tensorboard_array