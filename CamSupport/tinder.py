# Description: This file contains the TinderSession class, 
# which is used to create Dataset-Tinder sessions.
# The session is used to capture images from a camera and save 
# them to the correct class folder.

# =================================================================================================

from typing import Tuple
import PySimpleGUI as sg
from CamSupport.camctrl import CamHandler
from PIL import Image
import os
import keyboard

# =================================================================================================

class TinderSession():
    """Class to create Dataset-Tinder sessions
    """

    def __init__(self, out_path: str, classes: Tuple[str], tag: str, image_count: int = 0) -> None:
        """Create a new tinder session.

        Args:
            out_path (str): Folder path to save the images captured.
            classes (Tuple[str]): Two classes to classify the images.
            tag (str): Image tag used as a prefix for file names.
            image_count (int, optional): Start value of the image counter. Defaults to 0.

        Raises:
            Exception: If more than two classes are provided.
            Exception: If no classes are provided.
        """

        # Basic initialization
        self.image_count = image_count
        self.tag = tag

        # Dictionary to store the key pressed status
        # This is used to prevent holding down the key to save multiple images 
        self.KEY_PRESSED = {
            "left arrow": False,
            "right arrow": False,
            "space": False,
        }

        # If it doesn't exist, create the output directory
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # Check if the number of classes is correct
        if len(classes) > 2:
            raise Exception("Tinder only supports binary classification at the moment.")
        elif len(classes) == 0:
            raise Exception("No classes provided.")
        
        # Create a directory for each class
        for class_name in classes:
            if not os.path.exists(os.path.join(out_path, class_name)):
                os.makedirs(os.path.join(out_path, class_name))
        
        # Save the classes
        self.classes = classes

        # Create a dictionary to store the output paths
        self.out_dict = {class_name: os.path.join(out_path, class_name) for class_name in classes}
        self.out_dict.update({"root": out_path})

        # Initialize the camera handler
        self.cam = CamHandler()

    def save_image(self, image: Image.Image, class_name: str, mode: str = "normal") -> None:
        """Save an image to the correct class folder.

        Args:
            image (Image.Image): Image to save.
            class_name (str): Classification of the image.
            mode (str, optional): Save mode. Defaults to "normal".

        Raises:
            Exception: If mode is not "normal" or "mock".

        In "normal" mode, the image is saved to the output folder and the file name is printed to the console along with its class.
        In "mock" mode, the image is not saved, but the file name is printed to the console along with its class.
        """

        # Check if the mode is valid
        if mode not in {"normal", "mock"}:
            raise Exception("Invalid mode")
        
        # Create the image file name and path
        file_name = f"{self.tag}_{self.image_count:05}.png"
        file_path = os.path.join(self.out_dict[class_name], file_name)

        # Save the image if the mode is "normal"
        if mode == "normal":
            image.save(file_path)
            print(f"{file_name}: {class_name}")
        else:
            print(f"{file_name}: {class_name}")

        # Increment the image counter
        self.image_count += 1

    def key_pressed(self, key: str) -> bool:
        """Check if on of the relevant keys has been pressed.

        Args:
            key (str): Key to check.

        Raises:
            Exception: If the key is not in the list of relevant keys.

        Returns:
            bool: Whether the key is pressed or not.

        Pressing the key will return True only once until the key is released and pressed again.
        """
        
        # Check if the key is valid
        if key not in self.KEY_PRESSED.keys():
            raise Exception("Invalid key")
        
        # Check if the key is pressed
        if keyboard.is_pressed(key):

            # Check if the key was not pressed before
            if not self.KEY_PRESSED[key]:

                # Update the key status and return True
                self.KEY_PRESSED[key] = True
                return True
            
        # If the key was pressed before, return False
        # This is used to prevent holding down the key to save multiple images
        else: 
            self.KEY_PRESSED[key] = False
            return False

    def start_gui(self) -> None:
        """Start the Dataset-Tinder session using the OS window.
        """

        # Create the layout for the window
        # The window contains an image preview and two buttons to classify the image
        # The "Update Preview" button updates the image preview with the current camera 
        # image, but does not save it
        layout = [
            [sg.Column([[sg.Image(data='', key='image')]], justification='center')],
            [
                sg.Button(
                    self.classes[0], 
                    button_color=('green', sg.COLOR_SYSTEM_DEFAULT), 
                    border_width=0,
                    size=(25, 1)
                ),
                sg.Push(),
                sg.Button("Update Preview"),
                sg.Push(),
                sg.Button(
                    self.classes[1], 
                    button_color=('red', sg.COLOR_SYSTEM_DEFAULT), 
                    border_width=0,
                    size=(25, 1)
                )
            ]
        ]

        window = sg.Window('Dataset Tinder', layout, finalize=True)

        # Makes the window update the preview once at the beginning of a session
        initial_run = True

        # Store the current image to prevent multiple captures
        current_image = None

        while True:
            event, values = window.read(timeout=50)

            if event == sg.WINDOW_CLOSED:
                break
            
            if initial_run:
                # Capture the first image and display it
                current_image = self.cam.get_image()
                current_image_scaled = self.cam.get_image(factor=0.5)
                img_bytes = CamHandler.pil_to_bytes(current_image_scaled)
                window['image'].update(data=img_bytes)
                initial_run = False

            # Check if any of the relevant keys are pressed
            left_pressed = self.key_pressed("left arrow")
            right_pressed = self.key_pressed("right arrow")
            space_pressed = self.key_pressed("space")

            any_pressed = left_pressed or right_pressed or space_pressed

            # Save the image to the correct class -> First Class
            if event == self.classes[0] or left_pressed:
                self.save_image(current_image, self.classes[0], mode="normal")

            # Save the image to the correct class -> Second Class       
            if event == self.classes[1] or right_pressed:
                self.save_image(current_image, self.classes[1], mode="normal")

            # Update the preview with the current camera image
            # Is done automatically after each save or when the "Update Preview" button is pressed
            if event in {"Update Preview", self.classes[0], self.classes[1]} or any_pressed:
                current_image = self.cam.get_image()
                current_image_scaled = self.cam.get_image(factor=0.5)
                img_bytes = CamHandler.pil_to_bytes(current_image_scaled)
                window['image'].update(data=img_bytes)

        window.close()