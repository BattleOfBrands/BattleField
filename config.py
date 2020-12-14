
BRAND_NAMES = ["paytm", "ceat", "unacademy", "cred", "altroz", "dream11"]

DATASET = "/Users/hareesh/Timbuctoo/BattleOfBrands/dataset/match/*.jpg"
LOGOS_PATH = "tests/test_data/logos/"
SAVE_TO = "report.json"
PREDICTED_LOGO_PATH = "images/"
ITERATION_NAME = "test"

# DATASET = "/content/drive/My Drive/BattleofBrands/Dataset/Match/finals/first_half/*.jpg"
# LOGOS_PATH = "/content/drive/My Drive/BattleofBrands/Dataset/logos/predicted/train_data/"
# SAVE_TO = "/content/drive/My Drive/BattleofBrands/Dataset/logos/10_1.json"
# PREDICTED_LOGO_PATH = "/content/drive/My Drive/BattleofBrands/Dataset/logos/predicted/"
# ITERATION_NAME = "10_image_1"

# DATASET = "/content/drive/My Drive/BattleofBrands/Dataset/Match/finals/second_half/*.jpg"
# LOGOS_PATH = "/content/drive/My Drive/BattleofBrands/Dataset/logos/predicted/train_data/"
# SAVE_TO = "/content/drive/My Drive/BattleofBrands/Dataset/logos/10_2.json"
# PREDICTED_LOGO_PATH = "/content/drive/My Drive/BattleofBrands/Dataset/logos/predicted/"
# ITERATION_NAME = "10_image_2"

THRESHOLD = 0.65
MAX_LOGOS_PER_IMAGE = 10

SAVE_LOGO_PREDICTIONS = False
SAVE_IMAGE_PREDICTIONS = True
SHOW_PREDICTIONS = False

INPUT_TARGET_SIZE = 1000

WRITE_BATCH_SIZE = 50

RANDOMIZE_INPUT_IMAGES = False
RANDOM_SIZE_INPUT_IMAGES = 100

RANDOMIZE_INPUT_LOGOS = False
MAX_LOGOS_PER_CLASS = 5


TOTAL_ITERATIONS = 1