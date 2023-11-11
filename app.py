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
if 'picamera2' in sys.modules:
    from picamera_image_provider import PicameraImageProvider
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

        # configure style
        self.big_font = ('Helvetica', 15)
        self.style = ttk.Style()
        self.style.configure('TButton', font=self.big_font)
        self.style.configure('TNotebook.Tab', font=self.big_font)

        # create the main canvas for the image
        self.canvas = tk.Canvas(self.window, width=self.image_display_width, height=self.display_height)
        self.canvas.grid(row=0, column=0, padx=10, pady=5)

        # right frame for the controls
        self.tab_control = ttk.Notebook(self.window, width=int(self.display_width - self.image_display_width), height=self.display_height)
        self.tab_control.grid(row=0, column=1, padx=10, pady=5)

        self.provider_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.provider_tab, text='Image provider')
        self.detector_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.detector_tab, text='Object detection')
        self.classes_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.classes_tab, text='Classes')

        self.provider_buttons = []
        self.detector_buttons = []
        self.classes_buttons = []

        # set up image provider lists
        self.image_providers = []
        self.object_detectors = []

        self.selected_provider = -1
        self.selected_detector = -1

    def update_provider_controls(self, change_selection = -1):
        for button in self.provider_buttons:
            button.destroy()
        self.provider_buttons.clear()

        if change_selection >= 0:
            self.selected_provider = change_selection

        i = 0
        for provider in self.image_providers:
            button = ttk.Button(self.provider_tab, text=provider.name, command=lambda i=i: self.update_provider_controls(i))
            button.pack(fill=tk.X, padx=10, pady=0)
            self.provider_buttons.append(button)
            i += 1

    def update_detector_controls(self, change_selection = -1):
        for button in self.detector_buttons:
            button.destroy()
        self.detector_buttons.clear()

        if change_selection >= 0:
            self.selected_detector = change_selection
            self.update_classes_controls()

        i = 0
        for detector in self.object_detectors:
            button = ttk.Button(self.detector_tab, text=detector.name, command=lambda i=i: self.update_detector_controls(i))
            button.pack(fill=tk.X, padx=10, pady=0)
            self.detector_buttons.append(button)
            i += 1

    def update_classes_controls(self, toggle_class = -1):
        for button in self.classes_buttons:
            button.destroy()
        self.classes_buttons.clear()

        if toggle_class >= 0:
            self.object_detectors[self.selected_detector].enable_classes[toggle_class] = \
            not self.object_detectors[self.selected_detector].enable_classes[toggle_class]

        i = 0
        for detection_class in self.object_detectors[self.selected_detector].classes().items():
            button = tk.Button(self.classes_tab, text=detection_class[0], font=self.big_font, command=lambda i=i: self.update_classes_controls(i))
            if self.object_detectors[self.selected_detector].enable_classes[i]:
                button.configure(bg=detection_class[1])
            button.pack(fill=tk.X, padx=10, pady=0)
            self.classes_buttons.append(button)
            i += 1

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

# start updating parts of the main window
main_window.update_provider_controls()
main_window.update_detector_controls()
main_window.update_image()

# start the application
root.mainloop()
