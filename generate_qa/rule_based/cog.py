import os
import json
import random
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import constants


DOTA_v2_0_1_obj = "/home/anxiao/Datasets/MIGRANT/DIOR-R/label_1.json"
DIOR_R_1_obj = "/home/anxiao/Datasets/MIGRANT/DIOR-R/label_1.json"
cog_save_json = "/home/anxiao/Datasets/MIGRANT/cog.json"

with open(DOTA_v2_0_1_obj, 'r') as f:
    DOTA_v2_0_1_obj = json.load(f)
with open(DIOR_R_1_obj, 'r') as f:
    DIOR_R_1_obj = json.load(f)

cog_data = DOTA_v2_0_1_obj + DIOR_R_1_obj

image_item_dict = {}
for category in constants.DOTA_DIOR_COMBINE:
    if category not in image_item_dict:
        image_item_dict[category] = []
    for item in cog_data:
        if item['objects'][0]["object_name"] == category:  # only one object in one image
            image_item_dict[category].append(item)
    
num_images = [2, 3, 4]
cog_qa = []

for category in tqdm(image_item_dict):
    while len(image_item_dict[category]) >= 4:
        selected_item = random.sample(image_item_dict[category], random.choice(num_images))
        qa_item = {
            "category": category,
            "images": [item["image_path"] for item in selected_item],
            "question": "What common object can be found in the selected images?",
            "answer": f"The common object all these images share is a {category}.",
            "bboxes": [{k: v for k, v in item["objects"][0].items() if k != "object_name"} for item in selected_item]
        }
        cog_qa.append(qa_item)
        for used_item in selected_item:
            image_item_dict[category].remove(used_item)

with open(cog_save_json, 'w') as f:
    json.dump(cog_qa, f, indent=4)
