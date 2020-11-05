from random import randint
import glob
from logo_scout.os2d.few_shot_detection import FewShotDetection
from logo_scout.os2d.os2d.utils import visualization
from logo_scout.os2d.os2d.structures.bounding_box import cat_boxlist, BoxList

import logging
import json
import torch
BRAND_NAMES = ["dream11"] #, "paytm", "cred", "unacademy", "altroz"
DATASET = "/Users/hareesh/Timbuctoo/BattleOfBrands/dataset/logos/training/"
LOGOS_PATH = "/Users/hareesh/Timbuctoo/BattleOfBrands/dataset/logos/training/"

class ImageProcessor:
    def __init__(self):
        self.logo_identifiers = self.set_up(BRAND_NAMES)
        self.data_set = DATASET

    def set_up(self, brand_names):
        identifier = dict()
        for brand_name in brand_names:
            logo_paths = glob.glob( LOGOS_PATH + brand_name + "/*.png")
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
        return {"Hello": "World"}
        # if image is None:
        #     image = glob.glob("/Users/hareesh/Timbuctoo/BattleOfBrands/dataset/match/*.jpg")[1]
        #
        # for brand_name in self.logo_identifiers:
        #     boxes = self.logo_identifiers[brand_name].identify_logos(image)
        #     response[brand_name] = self.get_bounding_boxes(boxes)
        #
        # return response

    def start_processor(self):
        images = glob.glob(DATASET+"/*.jpg")
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

        with open('test.json') as f:
            file_data = json.load(f)

        file_data.update(data)

        with open('test.json', 'w') as f:
            json.dump(file_data, f)


image_processor = ImageProcessor()
image_processor.start_processor()