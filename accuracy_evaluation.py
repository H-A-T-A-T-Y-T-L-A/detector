import csv
import cv2
import os
from pathlib import Path

class AccuracyEvaluator:

    def __init__(self, annotation_folder, video_image_provider):
        self.video_image_provider = video_image_provider
        video_path = video_image_provider.video_path
        self.video_path = video_path;
        self.annotation_directory_path = annotation_folder;

        if not os.path.exists(self.annotation_directory_path):
            return

        self.annotation = None
        for annotation_file_path in os.listdir(self.annotation_directory_path):
            if Path(annotation_file_path).stem == Path(video_path).stem:
                annotation_file_path = os.path.join(self.annotation_directory_path, annotation_file_path)
                with open(annotation_file_path, newline='') as annotation_file:
                    reader = csv.reader(annotation_file, delimiter=',', quotechar='\"')
                    self.annotation = list(reader)

        if self.annotation is None:
            return

        self.frame_count = video_image_provider.frame_count

        self.frames_per_annotation = float(self.frame_count) / len(self.annotation)

    def evaluate(self):
        if self.annotation is None:
            return None
        current_index = int(self.video_image_provider.current_frame / self.frames_per_annotation)
        if current_index >= len(self.annotation):
            current_index = len(self.annotation) - 1
        return (self.annotation[current_index])
