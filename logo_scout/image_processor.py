from random import randint
import glob
from logo_scout.os2d.few_shot_detection import FewShotDetection
from logo_scout.os2d.os2d.utils import visualization
from logo_scout.os2d.os2d.structures.bounding_box import cat_boxlist, BoxList

import logging
import torch
BRAND_NAMES = ["dream11"] #, "paytm", "cred", "unacademy", "altroz"

class ImageProcessor:
    def __init__(self):
        self.logo_identifiers = self.set_up(BRAND_NAMES)

    def set_up(self, brand_names):
        identifier = dict()
        for brand_name in brand_names:
            logo_paths = glob.glob("/Users/hareesh/Timbuctoo/BattleOfBrands/dataset/logos/training/" + brand_name + "/*.png")
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

image_processor = ImageProcessor()