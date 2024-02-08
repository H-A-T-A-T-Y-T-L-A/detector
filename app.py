import cv2
import time
import sys
import os
import copy
import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
from PIL import Image, ImageTk
from threading import Thread
from threading import Event

from video_image_provider import VideoImageProvider
try:
    from picamera_image_provider import PicameraImageProvider
except:
    pass
from object_detection import NoDetection
from yolo_object_detection import YoloObjectDetection
from object_tracking import NoTracking, CV2Tracking
from accuracy_evaluation import AccuracyEvaluator


class MainWindow():
    def __init__(self, window, display_width, display_height, image_aspect_ratio):
        # save the main window and some parameters
        self.window = window
        self.display_width = display_width
        self.display_height = display_height
        self.image_display_width = display_height * image_aspect_ratio
        self.interval = 10
        self.detector_frequency = 30
        self.frame_counter = 0

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

        # image provider settings tab
        self.provider_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.provider_tab, text='Image')
        self.provider_buttons = []

        # object detector settings tab
        self.detector_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.detector_tab, text='Detection')
        self.detector_buttons = []
        
        # object detector settings tab
        self.tracker_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tracker_tab, text='Tracking')
        self.tracker_buttons = []

        # init detector run frequency controls
        self.detector_frequency_control = ttk.Frame(self.detector_tab)
        self.detector_frequency_control.pack(fill=tk.BOTH, padx=10, pady=10)
        ttk.Label(self.detector_frequency_control, text="Frames per run: ").grid(row=0, column=0, padx=3, pady=0);
        ttk.Button(self.detector_frequency_control, text="-", command=lambda: self.change_detector_frequency(-1)).grid(row=0, column=1, padx=3, pady=0)
        self.detector_frequency_display = ttk.Label(self.detector_frequency_control)
        self.detector_frequency_display.grid(row=0, column=2, padx=3, pady=0)
        self.change_detector_frequency(0)
        ttk.Button(self.detector_frequency_control, text="+", command=lambda: self.change_detector_frequency(+1)).grid(row=0, column=3, padx=3, pady=0)

        # class settings tab
        self.classes_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.classes_tab, text='Classes')
        self.classes_buttons = []

        # set up component lists
        self.image_providers = []
        self.selected_provider = -1

        self.object_detectors = []
        self.selected_detector = -1

        self.object_trackers = []
        self.selected_tracker = -1
        self.detections = None

        self.accuracy_evaluators = {}

    # change detector run frequency relatively
    def change_detector_frequency(self, change):
        self.detector_frequency += change
        if self.detector_frequency <= 0:
            self.detector_frequency = 1
        self.detector_frequency_display.configure(text=f"{self.detector_frequency:02}")

    # update ui and selection of image providers
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

    # update ui and selection of object detectors
    def update_detector_controls(self, change_selection = -1):
        for button in self.detector_buttons:
            button.destroy()
        self.detector_buttons.clear()

        if change_selection >= 0:
            self.selected_detector = change_selection
            self.update_classes_controls()
            self.object_trackers[self.selected_tracker].reset()

        i = 0
        for detector in self.object_detectors:
            button = ttk.Button(self.detector_tab, text=detector.name, command=lambda i=i: self.update_detector_controls(i))
            button.pack(fill=tk.X, padx=10, pady=0)
            self.detector_buttons.append(button)
            i += 1

    # update ui and selection of object trackers
    def update_tracker_controls(self, change_selection = -1):
        for button in self.tracker_buttons:
            button.destroy()
        self.tracker_buttons.clear()

        if change_selection >= 0:
            self.selected_tracker = change_selection

        i = 0
        for tracker in self.object_trackers:
            button = ttk.Button(self.tracker_tab, text=tracker.name, command=lambda i=i: self.update_tracker_controls(i))
            button.pack(fill=tk.X, padx=10, pady=0)
            self.tracker_buttons.append(button)
            i += 1

    # update ui and toggles of detected classes
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

    # capture an image and apply detection / classification / tracking
    def update_image(self):
        # do not ask for an image if there is no provider or an invalid one is selected
        if not 0 <= self.selected_provider < len(self.image_providers):
            self.window.after(self.interval * 10, self.update_image)
            return

        # acquire a new image
        self.image = self.image_providers[self.selected_provider].next()
        self.image = cv2.cvtColor(cv2.resize(self.image, self.object_detectors[self.selected_detector].image_resolution()), cv2.COLOR_BGR2RGB)
        self.image_to_detect = copy.deepcopy(self.image)

        self.truth = None
        current_evaluator = self.accuracy_evaluators.get(self.image_providers[self.selected_provider]);
        if current_evaluator is not None:
            self.truth = current_evaluator.evaluate();

        # draw true boxes if available
        if self.truth is not None:
            true_class_name = self.truth[-1]
            true_box = np.int16(self.truth[0:4])

            center_point = true_box[0:2]
            rect_size = np.int16(true_box[2:4]/2)
            start_point = center_point - rect_size
            end_point = center_point + rect_size
            text_point = start_point + [0, 10] + [0, rect_size[1] * 2]

            start_point = tuple(start_point)
            end_point = tuple(end_point)
            text_point = tuple(text_point)
            color = (0, 0, 0)

            self.image = cv2.rectangle(self.image, start_point, end_point, color, 1)
            self.image = cv2.putText(
                self.image, f"{true_class_name}",
                text_point, cv2.FONT_HERSHEY_SIMPLEX,
                .5, color, 1, cv2.LINE_AA
            )

        # if a detector is selected and enough frames have passed since last detection, run detection
        if 0 <= self.selected_detector < len(self.object_detectors):
            if self.frame_counter % self.detector_frequency == 0:
                # run the selected detector
                boxes, colors, names, confidences = self.object_detectors[self.selected_detector].detect(self.image_to_detect, .2, .5)
                self.detections = [boxes, colors, names, confidences]
                if 0 <= self.selected_tracker < len(self.object_trackers):
                    self.object_trackers[self.selected_tracker].init(self.image_to_detect, boxes)
            elif 0 <= self.selected_tracker < len(self.object_trackers):
                if self.detections is not None:
                    self.detections[0] = self.object_trackers[self.selected_tracker].track(self.image_to_detect)



            # draw bounding boxes
            if self.detections is not None and self.detections[0] is not None:
                for i in range(len(self.detections[0])):
                    boxes = self.detections[0]
                    colors = self.detections[1]
                    names = self.detections[2]
                    confidences = self.detections[3]

                    if boxes[i] is None:
                        continue

                    center_point = np.int16(boxes[i][0:2])
                    rect_size = np.int16(np.int16(boxes[i][2:4])/2)
                    start_point = center_point - rect_size
                    end_point = center_point + rect_size
                    text_point = start_point - [0, 5]

                    start_point = tuple(start_point)
                    end_point = tuple(end_point)
                    text_point = tuple(text_point)
                    color = tuple([int(c) for c in colors[i]])

                    self.image = cv2.rectangle(self.image, start_point, end_point, color, 1)
                    self.image = cv2.putText(
                        self.image, f"{names[i]} {confidences[i]:.02}",
                        text_point, cv2.FONT_HERSHEY_SIMPLEX,
                        .5, color, 1, cv2.LINE_AA
                    )

        self.frame_counter += 1

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
        new_provider = VideoImageProvider(os.path.join("videos", video))
        main_window.image_providers.append(new_provider)
        main_window.accuracy_evaluators[new_provider] = AccuracyEvaluator("video annotations", new_provider)

# if picamera2 library is installed, load it as a provider
if 'picamera2' in sys.modules:
    main_window.image_providers.append(PicameraImageProvider((640, 640)))

# add empty option for detectors
main_window.object_detectors.append(NoDetection())

# add all yolo models in "yolo" directory, each in its own subdirectory containing model.onnx and classes.csv
main_window.object_detectors.extend(YoloObjectDetection.look_for_models())

# add different types of trackers
main_window.object_trackers.append(NoTracking())
trackers = ["CSRT", "KCF", "MOSSE"]
for t in trackers:
    main_window.object_trackers.append(CV2Tracking(t, (640, 640)))

# start updating parts of the main window
main_window.update_provider_controls()
main_window.update_detector_controls()
main_window.update_tracker_controls()
main_window.update_image()

# start the application
root.mainloop()
