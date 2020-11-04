HIGH_IMPACT_PERCENT = 0.25
MID_IMPACT_PERCENT = 0.5

HIGH_IMPACT_SCORE = 1
MID_IMPACT_SCORE = 0.75
LOW_IMPACT_SCORE = 0.5


class BrandImpact:
    def __init__(self, image):
        self.image_height = image.height
        self.image_width = image.width

    def is_inside_box(self, bounding_box, percent):
        """
        Check if at-least one point of the bounding box is present within the specified percentage

        :param percent: point +/- (point * percent)
        :param bounding_box:
        :return:
        """

        percent = percent / 2  # we will be adding and subtracting from the mid point

        if (self.image_height * (1 - percent) <= bounding_box['x'] <= self.image_height * (1 + percent)) and \
                (self.image_width * (1 - percent) <= bounding_box['y'] <= self.image_width * (1 + percent)):
            return True

        if (self.image_height * (1 - percent) <= bounding_box['x'] + bounding_box['w'] <= self.image_height * (
                1 + percent)) and \
                (self.image_width * (1 - percent) <= bounding_box['y'] + bounding_box['h'] <= self.image_width * (
                        1 + percent)):
            return True

        return False

    def position_score(self, bounding_box):
        """

        :param bounding_box:
        :return:
        """

        if self.is_inside_box(bounding_box, HIGH_IMPACT_PERCENT):
            return HIGH_IMPACT_SCORE

        if self.is_inside_box(bounding_box, MID_IMPACT_PERCENT):
            return MID_IMPACT_SCORE

        return LOW_IMPACT_SCORE

    def consumption_score(self, bounding_box):
        """

        :param bounding_box:
        :return:
        """
        return (bounding_box['w'] * bounding_box['h']) / (self.image_height * self.image_width)

    @staticmethod
    def diversion_score(brands):
        """
        if there are more than one logo there is slight penalty

        :param brands:
        :return:
        """
        return (101 - len(brands)) / 100

    def compute_impact(self, brands):
        """
        Total impact for a brand is measured as a function of
        position of the logo
        area consumed by the logo
        diversion penalty (if there are more than one logo there is slight penalty)
        visibility of the logo, (if its partly visible => partial score)

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
            bounding_box = brand['bounding_box']
            response[brand['name']] = self.position_score(bounding_box) \
                                      * self.consumption_score(bounding_box) \
                                      * self.diversion_score(brands) \
                                      * brand.get('visibility', 1)

        return response
