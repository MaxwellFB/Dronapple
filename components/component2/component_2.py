"""
Component 2 - Object detector and tracker
"""

from .yolov7.yolov7 import Yolov7Detection
from .tracker import Tracker


class Component2:
    def __init__(self):
        self.yolov7 = Yolov7Detection(path_model='models/yolov7-tiny-416-32-env2.pt', img_size=416, stride=32,
                                              conf_threshold=0.25, iou_threshold=0.45)
        self.tracker = Tracker(objective_classes=[0])

    def detect(self, obs):
        classes, confidences, boxes_detector = self.yolov7.detect(obs, box_normalize=True, box_scale=True)
        return classes, confidences, boxes_detector

    def track(self, object_positions, classes):
        idx_closest_apple, all_distances = self.tracker.get_distance_object(main_position=(0.45, 0.55),
                                                                            object_positions=object_positions,
                                                                            classes=classes)
        return idx_closest_apple, all_distances