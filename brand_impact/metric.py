

class BrandImpact:
    def __init__(self, image):
        self.image_height = image.height
        self.image_width = image.width

    def is_inside_box(self, bounding_box, sub_screen):
        """

        :param sub_screen:
        :param bounding_box:
        :return:
        """
        dummy = self.image_width
        if sub_screen['x'] < bounding_box.x < sub_screen['x'] + sub_screen['h']:
            if sub_screen['y'] < bounding_box.y < sub_screen['y'] + sub_screen['w']:
                return True

        if sub_screen['x'] < bounding_box.x + bounding_box.w < sub_screen['x'] + sub_screen['h']:
            if sub_screen['y'] < bounding_box.y + bounding_box.h < sub_screen['y'] + sub_screen['w']:
                return True

        return False

    def position_score(self, bounding_box):
        """

        :param bounding_box:
        :return:
        """
        # TODO define numbers or derive the numbers
        high_impact = {"x": 1, "y": 1, "h": 1, "w": 1}
        mid_impact = {"x": 1, "y": 1, "h": 1, "w": 1}
        low_impact = {"x": 1, "y": 1, "h": 1, "w": 1}

        if self.is_inside_box(bounding_box, high_impact):
            return 1

        if self.is_inside_box(bounding_box, mid_impact):
            return 0.75

        return 0.25

    def consumption_score(self, bounding_box):
        """

        :param bounding_box:
        :return:
        """
        return (bounding_box.w * bounding_box.h) / (self.image_height * self.image_width)

    @staticmethod
    def diversion_score(brands):
        """

        :param brands:
        :return:
        """
        return (101 - len(brands)) / 100

    def compute_impact(self, brands):
        """

        :param brands:
            "brands": [
                {
                    "name": "<brand_name>",
                    "rectangle": {
                        "x": int,
                        "y": int,
                        "w": int,
                        "h": int
                    }
                }
            ]
        :return:
            {
                "<brand_name>": float
            }
        """
        response = dict()
        for brand in brands:
            response[brand['name']] = 0.0
            # bounding_box = brand.bounding_box
            # response[brand.name] = self.position_score(bounding_box) \
            #                        * self.consumption_score(bounding_box) \
            #                        * self.diversion_score(brands)

        return response
