# Description: 
# Evaluate the training performance of different models.

# =================================================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

# =================================================================================================


def overview(folder_path):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    # Loop through each file in the folder
    for file_name in file_list:
        if file_name.endswith('.csv'):  
            file_path = os.path.join(folder_path, file_name)  
            df = pd.read_csv(file_path, sep=";")

            file_info = file_name.split(" - ")
            print(file_info[1], file_info[2], max(df["Validation Accuracy"]))

def plot_losses(ax, csv_path):
    # Load the data from CSV
    data = pd.read_csv(csv_path, sep=";")
    
    # Check if the necessary columns are in the dataframe
    required_columns = ['Epoch', 'Loss', 'Validation Loss']
    if not all(col in data.columns for col in required_columns):
        raise ValueError("CSV must contain 'Epoch', 'Loss', and 'Validation Loss' columns")

    # Plotting
    ax.plot(data['Epoch'], data['Loss'], label='Training Loss')
    ax.plot(data['Epoch'], data['Validation Loss'], label='Validation Loss')
    
    # Adding plot title and labels
    ax.set_title('Training and Validation Loss by Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

def plot_accuracy(ax, csv_path):
    # Load the data from CSV
    data = pd.read_csv(csv_path, sep=";")
    
    # Check if the necessary columns are in the dataframe
    required_columns = ['Epoch', 'Validation Accuracy']
    if not all(col in data.columns for col in required_columns):
        raise ValueError("CSV must contain 'Epoch' and 'Validation Accuracy' columns")

    # Plotting
    ax.plot(data['Epoch'], data['Validation Accuracy'], label='Validation Accuracy')
    
    # Adding plot title and labels
    ax.set_title('Validation Accuracy by Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()

    return max(data['Validation Accuracy'])

def plot_overview(path: str, mode: str, out_dir: Optional[str] = None):
    if not mode in {"show", "save", "both"}:
        raise ValueError("Mode must be show, save or both!")
    if mode in {"save", "both"} and out_dir is None:
        raise ValueError("When trying to save the plot, an output path is mandatory!")

    file_name = os.path.basename(path)
    file_info = file_name.split(" - ")
    epochs = file_info[1]
    model = file_info[2]
    lr = file_info[3]

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(9, 12))
    plt.subplots_adjust(hspace=0.3)
    
    print(path)
    plot_losses(ax1, path)
    max_val_acc = '%.2f'%(round(plot_accuracy(ax2, path), 4)*100)

    plt.suptitle(f"{model} ({epochs}) - LR: {lr} - Max Val Acc: {max_val_acc}%", fontweight='bold'), 

    if mode in {"save", "both"}:
        plt.savefig(os.path.join(out_dir, file_name.replace("csv", "jpg")), format='jpg', dpi=200)

    if mode in {"show", "both"}:
        plt.show()
    

    

in_dir = "Performances\ResNet\CSVs"
file_list = os.listdir(in_dir)
out_dir = "Performances\ResNet\Plots"

for path in file_list:
    
    plot_overview(os.path.join(in_dir, path), "save", out_dir)