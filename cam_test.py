import gxipy as gx
from PIL import Image
import os

os.add_dll_directory("C:\Program Files\Daheng Imaging\GalaxySDK\APIDll\Win64")

# Get the camera manager
device_manager = gx.DeviceManager()

# Get device info list
dev_num, dev_info_list = device_manager.update_device_list()
if dev_num == 0:
    print("No device found")
    exit()

# Open device
# Get the list of basic device information
strSN = dev_info_list[0].get("sn")
# Open the device by serial number
cam = device_manager.open_device_by_sn(strSN)
# Start acquisition
cam.stream_on()

# Get data
# num is the number of images acquired
num = 1
for i in range(num):
    # Get an image from the 0th stream channel
    raw_image = cam.data_stream[0].get_image()
    numpy_image = raw_image.get_numpy_array()

# Display and save the got mono image
image = Image.fromarray(numpy_image, 'L')
print(image.size)
image.show()
