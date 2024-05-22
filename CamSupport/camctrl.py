# Description: Test the camera stream using the Daheng Imaging SDK

# =================================================================================================

import gxipy as gx
import PySimpleGUI as sg
import os
import numpy as np
import io
from PIL import Image

# =================================================================================================

class CamHandler():
    """Handler object for the camera
    """

    # Add the path of the SDK's dll to the system path
    os.add_dll_directory("C:\Program Files\Daheng Imaging\GalaxySDK\APIDll\Win64")

    def pil_to_bytes(pil_image: Image.Image) -> bytes:
        """Convert a pillow image to bytes

        Args:
            pil_image (Image.Image): Pillow image to convert

        Returns:
            bytes: Byte representation of the image
        """
        img_io = io.BytesIO()
        pil_image.save(img_io, format='PNG')
        return img_io.getvalue()

    def __init__(self):
        """Creates a new instance of the CamHandler class to handle inputs of the camera

        Raises:
            Exception: Raises an exception if no camera is found
        """
        
        # Daheng Imaging SDK initialization
        self.device_manager = gx.DeviceManager()
        self.dev_num, self.dev_info_list = self.device_manager.update_device_list()

        # Set initial streaming status
        self.stream = False

        if self.dev_num == 0:
            raise Exception("No device found")

        # Get the device by serial number 
        strSN: str = self.dev_info_list[0].get("sn")
        self.cam = self.device_manager.open_device_by_sn(strSN)

    def start_stream(self):
        """Start the camera stream.
        Checks if the stream is already on, if not, starts the stream
        """
        if self.stream:
            return
        self.stream = True
        self.cam.stream_on()

    def stop_stream(self):
        """Stop the camera stream.
        Checks if the stream is already off, if not, stops the stream
        """
        if not self.stream:
            return
        self.stream = False
        self.cam.stream_off()

    def get_numpy_image(self) -> np.ndarray:
        """Get the image from the camera as a numpy array

        Returns:
            np.ndarray: numpy array of the image
        """
        raw_image = self.cam.data_stream[0].get_image()
        numpy_image = raw_image.get_numpy_array()
        return numpy_image

    def get_image_on_stram(self, factor: float = 1) -> Image.Image:
        """Gets a pillow image from the camera. The image can be resized by a factor.
        REQUIRES THE STREAM TO BE ON!

        Args:
            factor (float, optional): Scaling factor to resize the image. Defaults to 1.

        Returns:
            Image.Image: Current image from the camera
        """
        numpy_image = self.get_numpy_image()
        image = Image.fromarray(numpy_image, 'L')

        # Resize
        if factor != 1:
            width, height = image.size
            new_size = (int(width * factor), int(height * factor))
            image = image.resize(new_size)

        return image
    
    def get_image(self, factor: float = 1) -> Image.Image:
        """Gets a pillow image from the camera. The image can be resized by a factor.
        REQUIRES THE STREAM TO BE OFF!

        Args:
            factor (float, optional): Scaling factor to resize the image. Defaults to 1.

        Returns:
            Image.Image: Current image from the camera
        """        

        # Capture the Image
        self.start_stream()
        numpy_image = self.get_numpy_image()
        self.stop_stream()

        image = Image.fromarray(numpy_image, 'L')

        # Resize
        if factor != 1:
            width, height = image.size
            new_size = (int(width * factor), int(height * factor))
            image = image.resize(new_size)

        return image

    
    def preview(self):
        """Open a window to preview the camera stream. 
        Expect lots of lag and low frame rate, as this is not optimized for performance.
        """

        self.start_stream()

        # Layout definitions
        layout = [
            [sg.Image(data='', key='image')],
            [sg.Button('Exit')]
        ]

        window = sg.Window('Camera Viewer', layout, finalize=True)

        while True:
            event, values = window.read(timeout=100)  # Update image every 100 milliseconds

            if event == sg.WINDOW_CLOSED or event == 'Exit':
                break
            
            image = self.get_image_on_stram(factor=0.5)
            img_bytes = CamHandler.pil_to_bytes(image)
            window['image'].update(data=img_bytes)

        window.close()
        self.stop_stream()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.cam.close_device()

if __name__ == "__main__":
    with CamHandler() as cam:
        cam.preview()

