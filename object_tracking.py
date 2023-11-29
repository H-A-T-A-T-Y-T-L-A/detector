import cv2
from abc import ABC, abstractmethod

class ObjectTracking(ABC):

    name = ""
    
    @abstractmethod
    def init(self, image, boxes):
        pass

    @abstractmethod
    def track(self, image):
        pass

    @abstractmethod
    def reset(self):
        pass

class NoTracking(ObjectTracking):

    name = "none"

    def init(self, image, boxes):
        pass

    def track(self, image):
        pass

    def reset(self):
        pass

class CV2Tracking(ObjectTracking):

    tracker_types = {
        "CSRT" : cv2.TrackerCSRT_create,
        "KCF" : cv2.TrackerKCF_create,
        "MOSSE" : cv2.legacy.TrackerMOSSE_create
    }

    def __init__(self, tracker, resolution):
        self.name = tracker
        self.tracker_init = self.tracker_types[tracker]
        self.resolution = resolution
        self.trackers = []

    def init(self, image, boxes):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.trackers = [self.tracker_init() for box in boxes]

        for i in range(len(boxes)):
            self.trackers[i].init(image, boxes[i])

    def track(self, image):
        image = cv2.resize(image, self.resolution)
        boxes = []

        for tracker in self.trackers:
            (success, box) = tracker.update(image)

            if success:
                boxes.append(box)
            else:
                boxes.append(None)

        return boxes

    def reset(self):
        self.trackers.clear()
