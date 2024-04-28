# Description: 
# Evaluate the training performance of different models.

# =================================================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

# =================================================================================================

def plot_losses(ax, csv_path):
    # Load the data from CSV
    data = pd.read_csv(csv_path, sep=";")
    
    # Check if the necessary columns are in the dataframe
    required_columns = ['Epoch', 'Train Loss', 'Test Loss']
    if not all(col in data.columns for col in required_columns):
        raise ValueError("CSV must contain 'Epoch', 'Train Loss', and 'Test Loss' columns")

    # Plotting
    ax.plot(data['Epoch'], data['Train Loss'], label='Train')
    ax.plot(data['Epoch'], data['Test Loss'], label='Test')
    
    # Adding plot title and labels
    ax.set_title('Train and Test Loss by Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

def plot_accuracy(ax, csv_path):
    # Load the data from CSV
    data = pd.read_csv(csv_path, sep=";")
    
    # Check if the necessary columns are in the dataframe
    required_columns = ['Epoch', 'Test Accuracy']
    if not all(col in data.columns for col in required_columns):
        raise ValueError("CSV must contain 'Epoch' and 'Test Accuracy' columns")

    # Plotting
    ax.plot(data['Epoch'], data['Test Accuracy'], label='Test Accuracy')
    
    # Adding plot title and labels
    ax.set_title('Test Accuracy by Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()

    return max(data['Test Accuracy']), data['Test Accuracy'].iloc[-1]

def plot_overview(path: str, mode: str, out_dir: Optional[str] = None):
    if not mode in {"show", "save", "both"}:
        raise ValueError("Mode must be show, save or both!")
    if mode in {"save", "both"} and out_dir is None:
        raise ValueError("When trying to save the plot, an output path is mandatory!")

    file_name = os.path.basename(path)
    file_info = file_name.split(" - ")
    epochs = file_info[1]
    model = file_info[2]

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(9, 12))
    plt.subplots_adjust(hspace=0.3)
    
    print(path)
    plot_losses(ax1, path)

    max_test_acc, final_test_acc = plot_accuracy(ax2, path)

    max_test_acc = '%.2f'%(round(max_test_acc, 4)*100)
    final_test_acc = '%.2f'%(round(final_test_acc, 4)*100)

    plt.suptitle(f"{model} ({epochs}) - Test Acc: Max {max_test_acc}%, Final {final_test_acc}%", fontweight='bold'), 

    if mode in {"save", "both"}:
        plt.savefig(os.path.join(out_dir, file_name.replace("csv", "jpg")), format='jpg', dpi=200)

    if mode in {"show", "both"}:
        plt.show()

in_dir = "Performances\ResNet\CSVs"
file_list = os.listdir(in_dir)
out_dir = "Performances\ResNet\Plots"

for path in file_list:
    if not os.path.splitext(path)[1] == ".csv":
        continue
    
    plot_overview(os.path.join(in_dir, path), "save", out_dir)