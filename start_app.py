# Description: 
# This script is used to start the classification application. 
# It loads a model and starts the GUI.

# =================================================================================================

from productive import app
import torchvision.transforms as transforms

# =================================================================================================

if __name__ == '__main__':
    
    # Index to class dictionary
    index_to_class_dict = {0: "IO", 1: "NIO"}

    # Transformations necessary for the model
    transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create an application
    application = app.ClassifyApp(index_to_class_dict, transformations)

    # Start the GUI
    application.start_gui()


