# Description: 
# This script is used to start the classification application. 
# It loads a model and starts the GUI.

# =================================================================================================

from models.resnet import get_pt_model
from productive import app
import torch
import torchvision.transforms as transforms

# =================================================================================================

if __name__ == '__main__':

    # MODEL_STATE_PATH = "02_saved_models\ResNet34 - 0.0005 - Kurz - RR\model_state_dict.pth"
    MODEL_STATE_PATH = "02_saved_models\ResNet34 - 0.0005 - Kurz - RR & Offset\model_state_dict.pth"

    classifier, _ = get_pt_model("ResNet34", 1, "cpu")
    state_dict = torch.load(MODEL_STATE_PATH, map_location="cpu")
    classifier.load_state_dict(state_dict)

    index_to_class_dict = {0: "IO", 1: "NIO"}

    transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    application = app.ClassifyApp(classifier, index_to_class_dict, transformations)

    application.start_gui()


