import json

file_data = dict()
with open("brand_impact/result_first_half.json") as f:
    file_data = json.load(f)

brand_count = {"paytm": 0, "cred": 0, "unacademy": 0, "altroz": 0, "dream11": 0}
for key, value in file_data.items():
    for brand_name, boxes in value.items():
        brand_count[brand_name] = brand_count[brand_name] + len(boxes)

print(brand_count)

file_data = dict()
with open("brand_impact/result_second_half.json") as f:
    file_data = json.load(f)

brand_count = {"paytm": 0, "cred": 0, "unacademy": 0, "altroz": 0, "dream11": 0}
for key, value in file_data.items():
    for brand_name, boxes in value.items():
        brand_count[brand_name] = brand_count[brand_name] + len(boxes)

print(brand_count)

print("**"*20)
file_data = dict()
brand_count = {"paytm": 0, "cred": 0, "unacademy": 0, "altroz": 0, "dream11": 0}

with open("brand_impact/result_first_half.json") as f:
    file_data = json.load(f)

for key, value in file_data.items():
    for brand_name, boxes in value.items():
        brand_count[brand_name] = brand_count[brand_name] + len(boxes)

with open("brand_impact/result_second_half.json") as f:
    file_data = json.load(f)

for key, value in file_data.items():
    for brand_name, boxes in value.items():
        brand_count[brand_name] = brand_count[brand_name] + len(boxes)

print(brand_count)