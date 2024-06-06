# Description: 
# This file contains the TinderSession class, which is used to create Dataset-Tinder sessions.
# The session is used to capture images from a camera and save them to the correct class folder.

# =================================================================================================

import json
from typing import Tuple
import PySimpleGUI as sg
import traceback
import torch
import torchvision.transforms as transforms
from utils.camctrl import CamHandler
import keyboard
from models import (
    resnet,
    vgg,
    convnext,
    regnet,
    densenet
)

# =================================================================================================

class MissingClassifierException(Exception):
    """Raised when the classifier is not set yet but a classification is requested.
    """
    def __init__(self):
        super().__init__("Classifier not set yet!")

class ClassifyApp():
    """The ClassifyApp class is used to classify images from the camera using a loaded model.
    It also provides a GUI to interact with the application and load models.
    """

    def __init__(self, index_to_class_dict: dict, transformations: transforms.Compose):
        """Initializes the ClassifyApp object.

        Args:
            index_to_class_dict (dict): Dictionary mapping the index to the class name.
            transformations (transforms.Compose): Transformations to apply to the image before classification.
        """

        self.index_to_class_dict = index_to_class_dict
        self.trasnformations = transformations

        # Initialize the camera handler
        self.cam = CamHandler()

        # Initialize the key pressed dictionary to keep track of the key presses
        self.KEY_PRESSED = {
            "space": False
        }

        # Initialize the classifier and the current model
        self.current_model = None
        self.classifier = None

        # Load the saved models
        with open("saved_models.json", "r") as f:
            self.saved_models = json.load(f)

    def get_model_nicknames(self) -> list:
        """Returns a list of nicknames of the saved models.

        Returns:
            list: List of nicknames of the saved models.
        """
        return [model['nickname'] for model in self.saved_models["models"]]
    
    def load_model(self, model_data: dict) -> None:
        """Loads a model from the given data.

        Args:
            model_data (dict): Dictionary containing the model data.

        Raises:
            ValueError: If the model type is unknown.

        The model data should have the following keys:
        - `nickname`: Nickname of the model.
        - `type`: Type of the model (ResNet, VGG, ConvNeXt, RegNet, DenseNet).
        - `architecture`: Architecture of the model. (e.g. "ResNet18", "VGG11-BN")
        - `path`: Path to the saved model state dictionary.
        """

        # Extract the model data
        nickname = model_data['nickname']
        model_type = model_data['type']
        architecture = model_data['architecture']
        sd_path = model_data['path']

        # If the model is already loaded, return
        if self.current_model == nickname:
            return

        # Load the model depending on the type and architecture
        if model_type == "ResNet":
            self.classifier, _ = resnet.get_pt_model(architecture, 1, "cpu")
        elif model_type == "VGG":
            self.classifier, _ = vgg.get_pt_model(architecture, 1, "cpu")
        elif model_type == "ConvNeXt":
            self.classifier, _ = convnext.get_pt_model(architecture, 1, "cpu")
        elif model_type == "RegNet":
            self.classifier, _ = regnet.get_pt_model(architecture, 1, "cpu") 
        elif model_type == "DenseNet":
            self.classifier, _ = densenet.get_pt_model(architecture, 1, "cpu")
        else:
            raise ValueError("Unknown model type")

        # Load the model state dictionary from the path
        state_dict = torch.load(sd_path, map_location="cpu")
        self.classifier.load_state_dict(state_dict)

        # Set the current model
        self.current_model = nickname

    def classify_current_image(self) -> Tuple[str, float]:
        """Classify the current image from the camera.

        Raises:
            MissingClassifierException: When the classifier is not set yet.

        Returns:
            Tuple[str, float]: Class name and the sigmoid output of the classifier for the image.
        """

        if self.classifier is None:
            raise MissingClassifierException()

        # Get the image from the camera
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
        """Start the GUI for the classification application.
        """

        # Get the model nicknames
        nicknames = self.get_model_nicknames()

        # Set the theme of the GUI to Black
        sg.theme('Black')

        # Define the layout of the GUI
        layout = [
            [   
                # Area to select the model
                sg.Text('Select a model:'),
                sg.Listbox(values=nicknames, size=(30, 2), enable_events=True, key='model_list'),
                sg.Push(),
                sg.Button("Load Model"),
            ],
            [sg.HorizontalSeparator()],
            [
                # Are to display the current model, prediction and sigmoid output
                sg.Text("Current Model: Nan", key='current_model'),
                sg.Push(),
                sg.Text("Prediction: NaN", key='prediction'),
                sg.Push(),
                sg.Text("Sigmoid Out: NaN", key='output')
            ],
            [sg.HorizontalSeparator()],
            [
                # Area to display the camera preview
                sg.Column([[sg.Image(data='', key='image')]], justification='center')
            ],
            [sg.HorizontalSeparator()],
            [
                # Area to classify the image button
                sg.Push(),
                sg.Button("Classify"),
                sg.Push()
            ],
            [sg.HorizontalSeparator()]
        ]

        # Create the window
        window = sg.Window("Classify App", layout, finalize=True)

        # Makes the window update the preview once at the beginning of a session
        initial_run = True

        while True:
            event, values = window.read(timeout=50)

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

            # If the load model button is pressed, load the model
            if event == "Load Model":
                try:

                    # Get the selected model
                    selected_model = values['model_list'][0]

                    # Look for the model in the saved models and load it
                    for model in self.saved_models["models"]:
                        if model['nickname'] == selected_model:
                            self.load_model(model)
                            window['current_model'].update(f"Current Model: {selected_model}")
                            break

                except Exception as e :

                    # If an error occurs, show a popup and print the error
                    sg.popup_error("Something went wrong while loading the model...")
                    traceback.print_exc()

            if event == "Classify" or space_pressed:
                current_image_scaled = self.cam.get_image(factor=0.5)
                img_bytes = CamHandler.pil_to_bytes(current_image_scaled)
                window['image'].update(data=img_bytes)

                try:

                    class_name, sigmoid_out = self.classify_current_image()
                    sigmoid_out = round(sigmoid_out, 6)

                    if class_name == list(self.index_to_class_dict.values())[0]:
                        col = 'green'
                    else:
                        col = 'red'

                    window['prediction'].update(f"Prediction: {class_name}", text_color=col)
                    window['output'].update(f"Sigmoid Out: {sigmoid_out}")

                except MissingClassifierException:
                    sg.popup_cancel("Classifier not set yet!")

                except Exception as e:
                    sg.popup_error("Something went wrong while classifying the image...")
                    traceback.print_exc()

        window.close()