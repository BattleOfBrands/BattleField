from random import randint
import glob
from logo_scout.os2d.few_shot_detection import FewShotDetection
import logging

class ImageProcessor:
    def __init__(self):
        self.logo_identification = self.set_up("dream11")
        self.detect_logos()

    def set_up(self, brand_name):
        logo_paths = glob.glob("/Users/hareesh/Timbuctoo/BattleOfBrands/dataset/logos/training/" + brand_name + "/*.png")

        return FewShotDetection(logo_paths)

    def get_bounding_boxes(self, boxes):
        return boxes.bbox_xyxy

    def detect_logos(self, image=None):
        """

        :param image:
        :return:
        """
        image_paths = glob.glob("/Users/hareesh/Timbuctoo/BattleOfBrands/dataset/match/*.jpg")

        boxes = self.logo_identification.identify_logos(image_paths[1])
        print(self.get_bounding_boxes(boxes))

image_processor = ImageProcessor()