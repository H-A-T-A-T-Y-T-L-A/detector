import cv2
import time
import sys
import os
import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
from PIL import Image, ImageTk
from threading import Thread
from threading import Event

from video_image_provider import VideoImageProvider
#from picamera_image_provider import PicameraImageProvider
from object_detection import NoDetection
from yolo_object_detection import YoloObjectDetection


class MainWindow():
    def __init__(self, window, display_width, display_height, image_aspect_ratio):
        # save the main window and some parameters
        self.window = window
        self.display_width = display_width
        self.display_height = display_height
        self.image_display_width = display_height * image_aspect_ratio
        self.interval = 10

        # create the main canvas for the image
        self.canvas = tk.Canvas(self.window, width=self.image_display_width, height=self.display_height)
        self.canvas.grid(row=0, column=0, padx=10, pady=5)

        # right frame for the controls
        self.right_frame = ttk.Frame(self.window, width=(self.display_width - self.image_display_width), height=self.display_height)
        self.right_frame.grid(row=0, column=1, padx=10, pady=5)

        self.detector_selector = ttk.Combobox(self.right_frame, values=[], state='readonly')
        self.detector_selector.bind('<<ComboboxSelected>>', self.update_controls)
        self.detector_selector.pack(padx=5, pady=5)

        self.provider_selector = ttk.Combobox(self.right_frame, values=[], state='readonly')
        self.provider_selector.bind('<<ComboboxSelected>>', self.update_controls)
        self.provider_selector.pack(padx=5, pady=5)

        # set up image provider lists
        self.image_providers = []
        self.object_detectors = []

        self.selected_provider = -1
        self.selected_detector = -1

        # start rendering the image
        self.update_image()

    def update_controls(self, event=None):
        provider_names = [provider.name for provider in self.image_providers]
        self.provider_selector['values'] = provider_names
        selected_provider_name = self.provider_selector.get()
        if selected_provider_name in provider_names and selected_provider_name != '':
            self.selected_provider = provider_names.index(selected_provider_name)
        else:
            self.selected_provider = -1

        detector_names = [detector.name for detector in self.object_detectors]
        self.detector_selector['values'] = detector_names
        selected_detector_name = self.detector_selector.get()
        if selected_detector_name in detector_names and selected_detector_name != '':
            self.selected_detector = detector_names.index(selected_detector_name)
        else:
            self.selected_detector = -1

    def update_image(self):
        # do not ask for an image if there is no provider or an invalid one is selected
        if not 0 <= self.selected_provider < len(self.image_providers):
            self.window.after(self.interval * 10, self.update_image)
            return

        # acquire a new image
        self.image = self.image_providers[self.selected_provider].next()
        self.image = cv2.resize(self.image, self.object_detectors[self.selected_detector].image_resolution())

        # do not ask for an image if there is no provider or an invalid one is selected
        if 0 <= self.selected_detector < len(self.object_detectors):
            # run the selected detector
            boxes, colors, names, confidences = self.object_detectors[self.selected_detector].detect(self.image, .3, .5)
            # draw bounding boxes
            for i in range(len(boxes)):
                center_point = np.int16(boxes[i][0:2])
                rect_size = np.int16(np.int16(boxes[i][2:4]/2))
                start_point = center_point - rect_size
                end_point = center_point + rect_size
                text_point = start_point - [0, 5]

                start_point = tuple(start_point)
                end_point = tuple(end_point)
                text_point = tuple(text_point)
                color = tuple([int(c) for c in colors[i]])

                self.image = cv2.rectangle(self.image, start_point, end_point, color, 1)
                self.image = cv2.putText(
                    self.image, " ".join([names[i], str(confidences[i])]),
                    text_point, cv2.FONT_HERSHEY_SIMPLEX,
                    .5, color, 1, cv2.LINE_AA
                )

        # format the image to be displayed
        self.image = Image.fromarray(self.image)
        self.image = ImageTk.PhotoImage(self.image)

        # draw the image onto the GUI
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

        # update again after set interval
        self.window.after(self.interval, self.update_image)


# start the main window
root = tk.Tk()
main_window = MainWindow(root, display_width=1024, display_height=600, image_aspect_ratio=1/1)

# load all videos in directory "videos" as input providers
if os.path.exists("videos"):
    for video in os.listdir("videos"):
        main_window.image_providers.append(VideoImageProvider(os.path.join("videos", video)))

# if picamera2 library is installed, load it as a provider
if 'picamera2' in sys.modules:
    main_window.image_providers.append(PicameraImageProvider((640, 640)))

# add empty option for detectors
main_window.object_detectors.append(NoDetection())

# add all yolo models in "yolo" directory, each in its own subdirectory containing model.onnx and classes.csv
main_window.object_detectors.extend(YoloObjectDetection.look_for_models())

# update the controls to correctly show data
main_window.update_controls()

# start the application
root.mainloop()
