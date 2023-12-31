import time
import cv2
import numpy as np
from picamera2 import Picamera2, Preview
from libcamera import Transform, controls
from image_provider import ImageProvider

class PicameraImageProvider(ImageProvider):

    name = "Camera"
    last_time = time.time()
    current_time = time.time()

    def __init__(self, resolution):
        self.camera = Picamera2()
        self.config = self.camera.create_preview_configuration(main={"size": resolution,
                                                                     "format": "RGB888"},
                                                               queue=True)
        self.camera.configure(self.config)
        self.camera.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        self.camera.start_preview()
        self.camera.start()

    def next(self):
        img = self.camera.capture_array()
        self.last_time = self.current_time
        self.current_time = time.time()
        return img

    def dt(self):
        return self.current_time - self.last_time
