from random import randint


class ImageProcessor:
    def __init__(self):
        pass

    def detect_logos(self, image):
        """

        :param image:
        :return:
        """
        return [
            {
                "name": "Microsoft",
                "visibility": 100,
                "bounding_box": {
                    "x": randint(0, 1024),
                    "y": randint(0, 768),
                    "w": randint(20, 100),
                    "h": randint(20, 50)
                }
            }
        ]
