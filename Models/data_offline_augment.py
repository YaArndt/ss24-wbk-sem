import os
from PIL import Image
from typing import List

# ALREADY EXECUTED! DO NOT RUN AGAIN!
# exit()

def rotate_images(folder_path: str, degree_options: List[int]):
    """Used for offline augmentation. Turns images in a folder by a specified number of degrees and saves them with a new file name.

    Args:
        folder_path (str): Path to the image folder
        degrees (int): Degrees to rotate the images by
    """

    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    for degrees in degree_options:
        for image_file in image_files:
        
            # Open the image
            image_path = os.path.join(folder_path, image_file)
            image = Image.open(image_path)

            # Rotate the image
            rotated_image = image.rotate(degrees)

            # Create a new file name with the degree number attached
            new_file_name = f"{os.path.splitext(image_file)[0]}_R_{degrees}{os.path.splitext(image_file)[1]}"
            new_file_path = os.path.join(folder_path, new_file_name)

            # Save the rotated image with the new file name
            rotated_image.save(new_file_path)

            # Close the image
            image.close()

        print("Image rotation complete for folder: ", folder_path, " by ", degrees, " degrees.")

# Example usage
folder_paths = [r"Data\BilderNeu\IO", r"Data\BilderNeu\NIO"]
degrees = [90, 180, 270]

for folder_path in folder_paths:
    rotate_images(folder_path, degrees)

