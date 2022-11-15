import os
import random
import pathlib as pl

import cv2
import torch
import numpy as np

from models.experimental import attempt_load
from .utils.datasets import letterbox
from .utils.general import non_max_suppression, scale_coords


class Yolov7Detection:
    def __init__(self, path_model='best.pt', img_size=416, stride=32, conf_threshold=0.25, iou_threshold=0.45):
        """
        Args:
            path_model (str): Model path
            img_size (int): Input image size model
            stride (int): Input image stride
            conf_threshold (float): Confidence threshold detections
            iou_threshold (float): IoU threshold
        """
        self.img_size = img_size
        self.stride = stride
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.half = self.device == 'cuda'  # half precision only supported on CUDA

        # Load model
        root_dir = pl.Path(__file__).resolve().parent
        self.model = attempt_load(pl.Path(root_dir, path_model), map_location=self.device)  # load FP32 model

        if self.half:
            self.model.half()  # to FP16

        if self.device != 'cpu':
            self.model(torch.zeros(1, 3, img_size, img_size).to(self.device).type_as(next(self.model.parameters())))

        self.LABELS = ['NR', 'NG', 'OR', 'OG']
        # Generate random colors for the bounding boxes
        self.COLORS = []
        # Random bounding boxes colors
        #for n in range(len(self.LABELS)):
        #    self.COLORS.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
        # BGR
        self.COLORS.append([0, 0, 255])
        self.COLORS.append([0, 255, 0])
        self.COLORS.append([255, 0, 255])
        self.COLORS.append([255, 0, 0])

    def detect(self, image, box_normalize=False, box_scale=False):
        """
        Args:
            image (uint8 BGR): Image to detect perforation
            box_normalize (bool): If the return of the boxes coordinates will be normalized or not
            box_scale (bool): If the return of the boxes coordinates will be scaled or not (coordinates for original
             image size)

        Returns:
            tuple[list[int], list[float], list[float, float, float, float]]:
                output_classes: ID class detected
                output_confidences: Confidences (0.0 - 1.0)
                output_boxes: Coordinates bounding box (X, Y, W, H)
        """
        original_img = image.copy()

        img_preprocessed = self._preprocess_image(image)

        pred = self.model(img_preprocessed, augment=False)[0]
        pred = pred.cpu()

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_threshold, self.iou_threshold)


        # (X, Y, W, H, Conf, Class)
        pred_detached = pred[0].detach().numpy()

        # To use the same scale that YOLOv4OpenCV
        if box_scale:
            pred_detached[:, :4] = scale_coords(img_preprocessed.shape[2:], pred[0][:, :4].clone(), original_img.shape)

        output_classes = []
        output_confidences = []
        output_boxes = []
        for (box_x, box_y, box_w, box_h, confidence, class_id) in pred_detached:
            # If want to normalize the coordinates of the boxes
            if box_normalize:
                h, w = image.shape[:2]
                # (X, Y, W, H)
                output_boxes.append([box_x / w, box_y / h, box_w / w, box_h / h])
            else:
                # (X, Y, W, H)
                output_boxes.append([box_x, box_y, box_w, box_h])
            output_classes.append(int(class_id))
            output_confidences.append(float(confidence))

        # Crop bounding box
        #if len(pred_detached) > 0:
        #    idx_argmax_conf = np.argmax(pred_detached[:, 4])
        #    cropped, bbox = self._crop_bbox(img_preprocessed, original_img, pred, idx_argmax_conf)

        return output_classes, output_confidences, output_boxes

    def _preprocess_image(self, img):
        """
        Preprocess image

        Args:
            img (uint8 BGR): Image to preprocess

        Returns:
            uint8 BGR: Image processed

        """
        # Padded resize
        img = letterbox(img, self.img_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to [3, img_size, img_size]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = img.unsqueeze(0)
        return img

    @staticmethod
    def _crop_bbox(img_preprocessed, original_image, pred, idx_argmax_conf):
        """
        Crop bbox coordinates

        Args:
            img_preprocessed (uint8 BGR): Image preprocessed for YOLOv7 input
            original_image (uint8 BGR): Original image BGR
            pred (Tensor): Output YOLOv7 prediction
            idx_argmax_conf (int): Bounding box index to crop

        Returns:
            tuple[uint8 BGR, tuple(int, int, int, int)]:
                Original image BGR cropped (bbox)
                Coordinates bounding

        """
        pred = pred[0]
        # Rescale boxes from img_preprocessed to original_image size
        pred[:, :4] = scale_coords(img_preprocessed.shape[2:], pred[:, :4].clone(), original_image.shape)

        x1, y1, w, h = int(pred[idx_argmax_conf][0]), int(pred[idx_argmax_conf][1]),\
                       int(pred[idx_argmax_conf][2]), int(pred[idx_argmax_conf][3])
        return original_image[y1: h, x1: w], (x1, y1, w, h)
        # Save cropped image in disk
        #cropped_images = temp[0]
        #from PIL import Image
        #Image.fromarray(cropped_images).save('crop.png')
        #return temp

    def draw_box(self, image, classes, confidences, boxes_coordinates, box_normalized=False, box_scaled=False,
                 image_shape_preprocessed=(256, 416), write_label=True):
        """
        Draw bounding boxes with the class name and confidence

        Args:
            image (uint8 BGR): Image to draw bounding boxes
            classes (list[int]): List with classes
            confidences (list[float]): List with confidences
            boxes_coordinates (list[float, float, float, float]): List with boxes coordinates (X, Y, W, H)
            box_normalized (bool): If the boxes coordinates are normalized or not
            box_scaled (bool): Uf the boxes are scaled
            image_shape_preprocessed (tuple[int, int]): Image shape input model (YOLO)
            write_label (bool): If is to write label on bounding boxes

        Returns:
            uint8: Image BGR with bbox
        """
        img = image.copy()
        boxes = boxes_coordinates.copy()


        if box_normalized:
            h, w = img.shape[:2]
            for idx in range(len(boxes)):
                boxes[idx] = [int(boxes[idx][0] * w), int(boxes[idx][1] * h), int(boxes[idx][2] * w),
                              int(boxes[idx][3] * h)]

        if not box_scaled:
            boxes = scale_coords(image_shape_preprocessed, torch.Tensor(boxes).clone(), image.shape)

        for (class_id, confidence, box) in zip(classes, confidences, boxes):
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          color=self.COLORS[class_id], thickness=1)

            if write_label:
                text = '%s: %.2f' % (self.LABELS[class_id], confidence)
                cv2.putText(img, text, (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color=self.COLORS[class_id], thickness=1)

        return img
