import cv2
from image_provider import ImageProvider


class VideoImageProvider(ImageProvider):

    video_path = ""
    frame_count = 0
    current_frame = 0
    video = None

    def __init__(self, video):
        self.video_path = video
        self.video = cv2.VideoCapture(video)

        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0

        self.name = 'video: ' + video


    def next(self):
        success, image = self.video.read()

        if (success):
            self.current_frame += 1
            return image

        self.reset()
        return self.next()

    def dt(self):
        if (self.video is None):
            return -1

        fps = self.video.get(cv2.CAP_PROP_FPS)
        return 1/fps

    def reset(self):
        self.current_frame = 0
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
