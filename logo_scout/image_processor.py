from random import randint

def process_image(image):
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
