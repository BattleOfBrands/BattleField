import unittest
from logo_scout.image_processor import ImageProcessor

TEST_DATA_PATH = "tests/test_data/integration_data/"


class MyTestCase(unittest.TestCase):
    def test_detect_logos(self):
        images = ["altroz.png", "cred.png", "dream11.png", "paytm.png", "unacademy.png"]
        image_processor = ImageProcessor()

        for image in images:
            detected_logos = image_processor.detect_logos(image)
            detected_logo_names = [logo['name'] for logo in detected_logos]
            brand_name = image.split(".")[0]
            self.assertIn(brand_name, detected_logo_names)


if __name__ == '__main__':
    unittest.main()
