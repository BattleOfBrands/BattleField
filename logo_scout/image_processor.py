from random import randint
import glob
from logo_scout.few_shot_learning import FewShotDetection
from logo_scout.os2d.os2d.utils import visualization
from logo_scout.os2d.os2d.structures.bounding_box import cat_boxlist, BoxList

import logging
import json
import torch
BRAND_NAMES = ["dream11"] #, "paytm", "cred", "unacademy", "altroz"
DATASET = "dataset/match/*.jpg"
LOGOS_PATH = "dataset/logos/training/"
SAVE_TO = "summary.json"

class ImageProcessor:
    def __init__(self, brand_names, dataset, save_to, logos_path):
        self.logo_identifiers = self.set_up(brand_names)
        self.data_set = dataset
        self.brand_names = brand_names
        self.save_to = save_to
        self.logos_path = logos_path

    def set_up(self, brand_names):
        identifier = dict()
        for brand_name in brand_names:
            logo_paths = glob.glob( self.logos_path + brand_name + "/*.png")
            identifier[brand_name] = FewShotDetection(logo_paths)
        return identifier

    def get_bounding_boxes(self, boxes, score_threshold=0.65, max_dets=8):
        scores = boxes.get_field("scores").clone()

        good_ids = torch.nonzero(scores.float() > score_threshold).view(-1)
        if good_ids.numel() > 0:
            if max_dets is not None:
                _, ids = scores[good_ids].sort(descending=False)
                good_ids = good_ids[ids[-max_dets:]]
            boxes = boxes[good_ids].cpu()
        else:
            boxes = BoxList.create_empty(boxes.image_size)

        boxes = boxes.bbox_xyxy

        # boxes = [list(box) for box in boxes]
        return boxes

    def detect_logos(self, image=None):
        """

        :param image:
        :return:
        """
        response = dict()

        for brand_name in self.logo_identifiers:
            boxes = self.logo_identifiers[brand_name].identify_logos(image)
            response[brand_name] = self.get_bounding_boxes(boxes)

        return response

    def start_processor(self):
        images = glob.glob(self.data_set)
        batch_size = 100
        buffer = dict()
        for image_path in images:
            buffer[image_path] = self.detect_logos(image_path)
            batch_size = batch_size - 1
            if batch_size == 0:
                self.write_to_json(buffer)
                batch_size = 100
                buffer = dict()
        self.write_to_json(buffer)

    def write_to_json(self, data):

        with open(self.save_to) as f:
            file_data = json.load(f)

        file_data.update(data)

        with open(self.save_to, 'w') as f:
            json.dump(file_data, f)


image_processor = ImageProcessor(brand_names=BRAND_NAMES, dataset=DATASET, save_to=SAVE_TO, logos_path=LOGOS_PATH)
image_processor.start_processor()