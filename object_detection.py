from abc import ABC, abstractmethod


class ObjectDetection(ABC):

    name = ""
    _image_height = 640
    _image_width = 640
    _classes = {}
    enable_classes = []

    def image_resolution(self):
        return (self._image_width, self._image_height)

    def classes(self):
        return {key: f'#{value[0]:02x}{value[1]:02x}{value[2]:02x}' for key, value in self._classes.items()}

    @abstractmethod
    def detect(self, image, threshold, nms_threshold):
        pass


class NoDetection(ObjectDetection):

    name = "none"

    def detect(self, image, threshold, nms_threshold):
        return [], [], [], []
