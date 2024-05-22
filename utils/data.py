# Description:
# Defines a set of data augmentation transformations to be applied to the images in the dataset.

# =================================================================================================

from PIL import Image, ImageOps
import random
from torch import Tensor
from torchvision.transforms import functional as F

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