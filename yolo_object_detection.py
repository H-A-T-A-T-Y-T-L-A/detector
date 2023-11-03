import cv2
import csv
import os
import numpy as np
from object_detection import ObjectDetection


class YoloObjectDetection(ObjectDetection):

    __net = None
    __class_names = None
    __class_colors = None

    def __init__(self, name, model, classes):
        self.name = name
        self.__net = cv2.dnn.readNetFromONNX(model)
        with open(classes, newline='') as classes_file:
            reader = csv.reader(classes_file, delimiter=',', quotechar='\"')

            self.__class_names = []
            self.__class_colors = []
            for row in reader:
                self.__class_names.append(row[0])
                self.__class_colors.append(tuple(
                    [int(str.replace(c, "\"", "")) for c in row[1:]]
                ))

    def detect(self, image, threshold, nms_threshold):
        image_blob = cv2.dnn.blobFromImage(
            image, scalefactor=1/255, size=(640, 640)
        )

        self.__net.setInput(image_blob)
        out = self.__net.forward()

        _, n_outputs, n_objects = out.shape

        class_ids = []
        class_confs = []
        boxes = []

        for i in range(n_objects):
            conf = out[:, 4:, i].reshape(-1)

            class_id = np.argmax(conf)
            class_conf = conf[class_id]

            if class_conf >= threshold:
                class_ids.append(class_id)
                class_confs.append(class_conf)
                boxes.append([int(a) for a in out[:, :4, i].reshape(4)])

        valid_boxes = cv2.dnn.NMSBoxes(
            boxes, class_confs, threshold, nms_threshold
        )

        return \
            np.int16(boxes)[valid_boxes], \
            np.array(self.__class_colors)[class_ids], \
            np.array(self.__class_names)[class_ids], \
            np.int16(class_confs)[valid_boxes]


    @staticmethod
    def look_for_models():
        base_folder = "yolo/"
        dir_list = os.listdir(base_folder)

        yolo_list = []
        for dir in dir_list:
            dir_path = os.path.join(base_folder, dir)

            if not os.path.isdir(dir_path):
                continue

            dir_items = os.listdir(dir_path)
            model_path = None
            class_list_path = None
            for item in dir_items:
                if item == "model.onnx":
                    model_path = os.path.join(base_folder, dir, item)
                elif item == "classes.csv":
                    class_list_path = os.path.join(base_folder, dir, item)

            if model_path == None or class_list_path == None:
                continue

            yolo_list.append(YoloObjectDetection('YOLO: ' + dir, model_path, class_list_path))


        return yolo_list
