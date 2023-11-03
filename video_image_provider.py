import cv2
from image_provider import ImageProvider


class VideoImageProvider(ImageProvider):

    __video = None
    __current_frame = 0

    def __init__(self, video):
        self.__video = cv2.VideoCapture(video)
        self.__current_frame = 0

        self.name = 'video: ' + video

    def next(self):
        self.__video.set(cv2.CAP_PROP_POS_FRAMES, self.__current_frame)
        success, image = self.__video.read()

        if (success):
            fps = self.__video.get(cv2.CAP_PROP_FPS)
            self.__current_frame += int(.3 * fps)
            return image

        self.__current_frame = 0
        return self.next()

    def dt(self):
        if (self.__video is None):
            return -1

        fps = self.__video.get(cv2.CAP_PROP_FPS)
        return 1/fps
