from logo_scout.image_processor import process_image
from brand_impact.metric import BrandImpact

from random import randint


class Game:
    def __init__(self):
        pass

    def create_

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
