from abc import ABC, abstractmethod


class ObjectDetection(ABC):

    name = ""
    _image_height = 640
    _image_width = 640

    def image_resolution(self):
        return (self._image_width, self._image_height)

    @abstractmethod
    def detect(self, image, threshold, nms_threshold):
        pass


class NoDetection(ObjectDetection):

    name = "none"

    def detect(self, image, threshold, nms_threshold):
        return [], [], [], []
