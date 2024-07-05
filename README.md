# Poltopfbefundung
Short notes on how to install the project code and its requirements

## Install the Daheng Imaging SDK
Visit the [vendors download page](https://www.get-cameras.com/customerdownloads) and download the [Windows SDK USB2+USB3+GigE V1 (including Directshow + Python) Galaxy V1.24.2308.9101](https://dahengimaging.com/downloads/Galaxy_Windows_EN_32bits-64bits_1.24.2308.9101.zip).
After installing the software, you should have a DLL directory at `C:\Program Files\Daheng Imaging\GalaxySDK\APIDll\Win64` or `C:\Program Files\Daheng Imaging\GalaxySDK\APIDll\Win32`. If the installation is located somwhere else, make sure to change the path in `utils\camctrl.py`.
## Setting up the Conda environment
This code is designed to work within a conda environment. Create a new conda environment **using python 3.11**. After that install the required packages from the `requirements.txt`. To do this run `pip install -r requirements.txt` (we know thats bad practice, but its the only way it worked).
