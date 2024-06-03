# Description: 
# This file contains the TinderSession class, which is used to create Dataset-Tinder sessions.
# The session is used to capture images from a camera and save them to the correct class folder.

# =================================================================================================

from typing import Tuple
import PySimpleGUI as sg
import torch
from torch import nn
import torchvision.transforms as transforms
from utils.camctrl import CamHandler
from PIL import Image
import os
import keyboard

# =================================================================================================


class ClassifyApp():

    def __init__(self, classifier: nn.Module, index_to_class_dict: dict, transformations: transforms.Compose):
        self.classifier = classifier
        self.index_to_class_dict = index_to_class_dict
        self.trasnformations = transformations
        self.cam = CamHandler()
        self.KEY_PRESSED = {
            "space": False
        }

    def classify_current_image(self) -> Tuple[str, float]:

        image = self.cam.get_image()

        image = self.trasnformations(image).unsqueeze(0)

        # Make a prediction
        with torch.no_grad():
            output = self.classifier(image).view(-1)
            sigmoid_out = torch.sigmoid(output)
            prediction = torch.round(sigmoid_out).int().item()

        return self.index_to_class_dict[prediction], sigmoid_out.item()

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

    def start_gui(self):

        sg.theme('Black')
        layout = [
            [sg.HorizontalSeparator()],
            [
                sg.Text("Prediction: NaN", key='prediction'),
                sg.Push(),
                sg.Text("Sigmoid Out: NaN", key='output')
            ],
            [sg.HorizontalSeparator()],
            [sg.Column([[sg.Image(data='', key='image')]], justification='center')],
            [sg.HorizontalSeparator()],
            [
                sg.Push(),
                sg.Button("Classify"),
                sg.Push()
            ],
            [sg.HorizontalSeparator()]
        ]

        window = sg.Window("Classify App", layout, finalize=True)
        # Makes the window update the preview once at the beginning of a session
        initial_run = True

        while True:
            event, _ = window.read(timeout=50)

            if event == sg.WINDOW_CLOSED:
                break
            
            if initial_run:
                # Capture the first image and display it
                current_image_scaled = self.cam.get_image(factor=0.5)
                img_bytes = CamHandler.pil_to_bytes(current_image_scaled)
                window['image'].update(data=img_bytes)
                initial_run = False

            # Check if the space key is pressed
            space_pressed = self.key_pressed("space")

            if event == "Classify" or space_pressed:
                current_image_scaled = self.cam.get_image(factor=0.5)
                img_bytes = CamHandler.pil_to_bytes(current_image_scaled)
                window['image'].update(data=img_bytes)

                class_name, sigmoid_out = self.classify_current_image()
                sigmoid_out = round(sigmoid_out, 6)

                if class_name == list(self.index_to_class_dict.values())[0]:
                    col = 'green'
                else:
                    col = 'red'

                window['prediction'].update(f"Prediction: {class_name}", text_color=col)
                window['output'].update(f"Sigmoid Out: {sigmoid_out}")

        window.close()