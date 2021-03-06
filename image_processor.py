from random import randint
import glob
import time
from logo_scout.few_shot_learning import FewShotDetection
from logo_scout.os2d.os2d.utils import visualization
from logo_scout.os2d.os2d.structures.bounding_box import cat_boxlist, BoxList

import random
import logging
import json
import torch
from config import *
import os

class ImageProcessor:
    def __init__(self, brand_names, dataset, save_to, logos_path):
        self.logo_identifiers = self.set_up(brand_names)
        self.data_set = dataset
        self.brand_names = brand_names
        self.save_to = save_to
        self.logos_path = logos_path
        self.create_dirs()

    def create_dirs(self):
        main_dir = PREDICTED_LOGO_PATH+ITERATION_NAME
        if not os.path.exists(main_dir):
            os.makedirs(main_dir)

        for brand_name in BRAND_NAMES:
            brand_dir = main_dir+"/"+brand_name
            if not os.path.exists(brand_dir):
                os.makedirs(brand_dir)

    def set_up(self, brand_names):
        identifier = dict()
        for brand_name in brand_names:
            print("Loading..", brand_name)
            logo_paths = glob.glob(LOGOS_PATH + brand_name + "/*.png")
            predicted_logo_paths = glob.glob(PREDICTED_LOGO_PATH + ITERATION_NAME + "/" + brand_name + "/*.png")
            logo_paths = logo_paths + predicted_logo_paths
            if RANDOMIZE_INPUT_LOGOS:
                logo_paths = [random.choice(logo_paths) for _ in range(MAX_LOGOS_PER_CLASS)]
            print("Found ", len(logo_paths), " logos")
            identifier[brand_name] = FewShotDetection(logo_paths, name=brand_name)
        return identifier

    def get_bounding_boxes(self, boxes, score_threshold=0.6, max_dets=8):
        scores = boxes.get_field("scores").clone()

        good_ids = torch.nonzero(scores.float() > score_threshold).view(-1)
        if good_ids.numel() > 0:
            if max_dets is not None:
                _, ids = scores[good_ids].sort(descending=False)
                good_ids = good_ids[ids[-max_dets:]]
            boxes = boxes[good_ids].cpu()
        else:
            boxes = BoxList.create_empty(boxes.image_size)

        default_boxes = boxes.get_field("default_boxes") if boxes.has_field("default_boxes") else None
        if default_boxes is not None:
            default_boxes = default_boxes[good_ids].cpu()

            # append boxes
            boxes = torch.cat([default_boxes.bbox_xyxy, boxes.bbox_xyxy], 0)
        else:
            boxes = boxes.bbox_xyxy

        return self.bounding_boxes(boxes)

    def bounding_boxes(self, boxes):
        bouding_boxes = list()
        for bounding_box in boxes:
            b_box = list()
            for cord in bounding_box:
                b_box.append(int(cord))
            bouding_boxes.append(b_box)
        return bouding_boxes

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

    def show_statistics(self):
        for brand_name in self.brand_names:
            input_logo_paths = glob.glob(LOGOS_PATH + brand_name + "/*.png")
            predicted_logo_paths = glob.glob(LOGOS_PATH + brand_name + "/*.png")
            print("Got ", len(predicted_logo_paths), "from ", len(input_logo_paths), "logos")

    def start_processor(self):
        images = glob.glob(self.data_set)  # #["tests/test_data/match_images/cred.png"]
        batch_size = 1
        buffer = dict()
        print("Total Images", len(images))
        if RANDOMIZE_INPUT_IMAGES == True:
            images = [random.choice(images) for _ in range(RANDOM_SIZE_INPUT_IMAGES)]

        start_time = time.time()
        print("Considered Images", len(images))
        completed = 0

        for image_path in images:
            completed = completed + 1
            buffer[image_path] = self.detect_logos(image_path)
            batch_size = batch_size - 1
            if batch_size == 0:
                print("Completed", completed / len(images))
                self.write_to_json(buffer)
                batch_size = WRITE_BATCH_SIZE
                buffer = dict()
        print("Time Taken:", time.time() - start_time)
        self.show_statistics()
        self.write_to_json(buffer)

    def write_to_json(self, data):
        try:
            with open(self.save_to) as f:
                file_data = json.load(f)
            file_data.update(data)

        except Exception as error:
            file_data = data
            print("JSON File not found.. creating and writing to new file")

        with open(self.save_to, 'w') as f:
            json.dump(file_data, f)


while TOTAL_ITERATIONS > 0:
    TOTAL_ITERATIONS = TOTAL_ITERATIONS - 1
    image_processor = ImageProcessor(brand_names=BRAND_NAMES, dataset=DATASET, save_to=SAVE_TO, logos_path=LOGOS_PATH)
    image_processor.start_processor()
