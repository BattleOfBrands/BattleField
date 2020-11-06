import os
import argparse
import logging
import random
import string
import matplotlib.pyplot as plt
import time

import torch
import torchvision.transforms as transforms

from logo_scout.os2d.os2d.modeling.model import build_os2d_from_config
from logo_scout.os2d.os2d.config import cfg
import logo_scout.os2d.os2d.utils.visualization as visualizer
from logo_scout.os2d.os2d.structures.feature_map import FeatureMapSize
from logo_scout.os2d.os2d.utils import setup_logger, read_image, get_image_size_after_resize_preserving_aspect_ratio

import random
import string
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision

from logo_scout.os2d.os2d.structures.feature_map import FeatureMapSize
from logo_scout.os2d.os2d.structures.bounding_box import cat_boxlist, BoxList
from logo_scout.os2d.os2d.config import cfg

logger = setup_logger("OS2D")

cfg.is_cuda = torch.cuda.is_available()
cfg.init.model = "models/os2d_v2-train.pth"
net, box_coder, criterion, img_normalization, optimizer_state = build_os2d_from_config(cfg)


class FewShotDetection:
    def __init__(self, logos_path):
        self.transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(img_normalization["mean"], img_normalization["std"])
        ])

        self.logos = self.load_logos(logos_path)

    def resize_image(self, image, target_size):
        h, w = get_image_size_after_resize_preserving_aspect_ratio(h=image.size[1],
                                                                   w=image.size[0],
                                                                   target_size=target_size)
        image = image.resize((w, h))
        return self.transform_image(image)

    def load_logos(self, logos_path):
        class_images_th = []
        for image_path in logos_path:
            class_image = read_image(image_path)
            class_image_th = self.resize_image(class_image, target_size=cfg.model.class_image_size)
            if cfg.is_cuda:
                class_image_th = class_image_th.cuda()

            class_images_th.append(class_image_th)
        if len(class_images_th) == 0:
            logging.error("Input logos not found?")
        return class_images_th

    def pre_process_image(self, image_path, target_size):
        image = read_image(image_path)
        input_image_th = self.resize_image(image, target_size=target_size)
        input_image_th = input_image_th.unsqueeze(0)
        if cfg.is_cuda:
            input_image_th = input_image_th.cuda()
        return input_image_th

    def identify_logos(self, image_path, target_size=1500):
        input_image_th = self.pre_process_image(image_path, target_size)
        with torch.no_grad():
            loc_prediction_batch, class_prediction_batch, _, fm_size, transform_corners_batch = net(
                images=input_image_th, class_images=self.logos)

        image_loc_scores_pyramid = [loc_prediction_batch[0]]
        image_class_scores_pyramid = [class_prediction_batch[0]]
        img_size_pyramid = [FeatureMapSize(img=input_image_th)]
        transform_corners_pyramid = [transform_corners_batch[0]]

        boxes = box_coder.decode_pyramid(image_loc_scores_pyramid, image_class_scores_pyramid,
                                         img_size_pyramid, [i for i in range(0, len(self.logos))],
                                         nms_iou_threshold=cfg.eval.nms_iou_threshold,
                                         nms_score_threshold=cfg.eval.nms_score_threshold,
                                         transform_corners_pyramid=transform_corners_pyramid)
        boxes.remove_field("default_boxes")
        cfg.visualization.eval.max_detections = 8
        cfg.visualization.eval.score_threshold = float(0.6)
        visualizer.show_detections(boxes, read_image(image_path), cfg.visualization.eval)

        return boxes
