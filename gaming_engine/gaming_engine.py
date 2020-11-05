from logo_scout.image_processor import ImageProcessor
from brand_impact.metric import BrandImpact
from selenium import webdriver
from random import randint
from PIL import Image

import cv2
import time
import json


class Game:
    def __init__(self, video=None, save_to="/tmp"):
        if video is None:
            self.web_driver = self.create_web_driver()
            self.begin_game(save_to)
        else:
            self.video_to_image(video, save_to)

    def video_to_image(self, video_path, save_to):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(5)
        count = 0
        while (cap.isOpened()):
            skip_frames = fps

            while(skip_frames > 0):
                skip_frames = skip_frames -1
                cap.read()

            ret, frame = cap.read()
            cv2.imwrite(save_to+'/frame{:d}.jpg'.format(count), frame)
            count = count + 1

        cap.release()
        cv2.destroyAllWindows()


    @staticmethod
    def create_web_driver():
        """

        :return:
        """
        # TODO executable path to be present here itself
        driver = webdriver.Firefox()

        # Opening the website
        driver.get("https://www.google.com")

        return driver

    def begin_game(self, save_to):
        count = 0
        while True:
            count = count + 1
            self.web_driver.save_screenshot(save_to+"/"+str(count)+".jpg")
            # image = Image.open("image.png")
            # print(image)
            # data = self.process_image(self.time_in_sec, "image.png")
            # print(data)
            # self.write_to_json(data)
            # cv2.resize(image, (250, 174))
            # cv2.imshow('Game', image)
            time.sleep(1)
            # cv2.destroyAllWindows()

    def calculate_impact(self, image, brands):
        """

        :param brands:
        :return:
        """
        brand_impact = BrandImpact(image)
        computed_brand_impact = brand_impact.compute_impact(brands)
        return computed_brand_impact

    def detect_logo(self, image):
        """

        :param image:
        :return:
        """
        image_processor = ImageProcessor()
        logos_identified = image_processor.detect_logos(image)
        return logos_identified

    def write_to_database(self):
        pass

    def process_image(self, id, image):
        """

        :param image:
        :return:
        """
        brands_identified = self.detect_logo(image)
        return {id: brands_identified}
        # brand_impact = self.calculate_impact(image, brands_identified)
        # self.write_to_database()
        # return brand_impact

    def write_to_json(self, data):

        with open('test.json') as f:
            file_data = json.load(f)

        file_data.update(data)

        with open('test.json', 'w') as f:
            json.dump(file_data, f)

game = Game(video="/Users/hareesh/Timbuctoo/BattleOfBrands/BattleField/video.mp4")
game
# game.begin_game()