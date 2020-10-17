from logo_scout.image_processor import ImageProcessor
from brand_impact.metric import BrandImpact
from selenium import webdriver
from random import randint

import cv2
import time


class Game:
    def __init__(self):
        self.web_driver = self.create_web_driver()

    def create_web_driver(self):
        """

        :return:
        """
        # TODO executable path to be present here itself
        driver = webdriver.Firefox()

        # Opening the website
        driver.get("https://www.google.com")

        return driver

    def begin_game(self):
        while True:
            self.web_driver.save_screenshot("image.png")
            image = cv2.imread("image.png")
            cv2.resize(image, (250, 174))
            cv2.imshow('Game', image)

            time.sleep(0.25)
            cv2.destroyAllWindows()

    def calculate_impact(self):
        brand_impact = BrandImpact()

    def detect_logo(self):
        image_processor = ImageProcessor()

    def process_image(self, image):
        """

        :param image:
        :return:
        """
        print(image)
        return [
            {
                "name": "Microsoft",
                "visibility": 100,
                "rectangle": {
                    "x": randint(0, 1024),
                    "y": randint(0, 768),
                    "w": randint(20, 100),
                    "h": randint(20, 50)
                }
            }
        ]
