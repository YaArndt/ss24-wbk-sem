# Description: 
# This script is used to create a dataset for the project.

# =================================================================================================

from datetime import datetime
import os
from utils.tinder import TinderSession

# =================================================================================================

# Define the output path and the directory name
OUT_PATH = "01_data_selfmade" 
DIR = "errors_lang"

# Setting the path
path = os.path.join(OUT_PATH, DIR)
date_str = datetime.now().strftime("%Y%m%d")

if __name__ == "__main__":

    session = TinderSession(out_path=path, classes=("IO", "NIO"), tag=f"{date_str}-{DIR}")
    session.start_gui()