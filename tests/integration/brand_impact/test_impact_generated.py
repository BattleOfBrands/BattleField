import unittest
from brand_impact.metric import BrandImpact


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.brand_impact = BrandImpact(h=1000, w=1000)

    def test_full_impact(self):
        # "brands": [
        #     {
        #         "name": "<brand_name>",
        #         "rectangle": {
        #             "x": int,
        #             "y": int,
        #             "w": int,
        #             "h": int
        #         }
        #     }
        # ]

        self.assertEqual(True, False)

    def test_zero_impact(self):
        self.assertEqual(True, False)

    def test_partial_impact(self):
        self.assertEqual(True, False)

    def test_brand_names(self):
        brands = [{"name": "brand_1", "bounding_box": {}}]

        computed_brand_impact = self.brand_impact.compute_impact(brands)

        # Test number of brands given as input is equal to the number of present in the ouput
        brands_input = brands.__len__()
        brands_output = computed_brand_impact.__len__()
        self.assertEqual(brands_input, brands_output)

        # # Test all the brands are returned
        for brand in brands:
            print(brand, brands_output)
            self.assertIn(brand['name'], computed_brand_impact)


if __name__ == '__main__':
    unittest.main()
