import os
import copy
import json
import random

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import constants, cog_query


DOTA_v2_0_1_obj = "/home/anxiao/Datasets/MIGRANT/DOTA-v2_0/label_1.json"
DIOR_R_1_obj = "/home/anxiao/Datasets/MIGRANT/DIOR-R/label_1.json"
cog_save_json = "/home/anxiao/Datasets/MIGRANT/sft/cog.json"
cog_save_txt = "/home/anxiao/Datasets/MIGRANT/stat_txt/cog.txt"
os.makedirs(os.path.dirname(cog_save_json), exist_ok=True)
os.makedirs(os.path.dirname(cog_save_txt), exist_ok=True)

with open(DOTA_v2_0_1_obj, 'r') as f:
    DOTA_v2_0_1_obj = json.load(f)
with open(DIOR_R_1_obj, 'r') as f:
    DIOR_R_1_obj = json.load(f)
cog_data = DOTA_v2_0_1_obj + DIOR_R_1_obj

image_item_dict = {cat: [] for cat in constants.DOTA_DIOR_COMBINE}
for item in cog_data:
    category = item['objects'][0]["object_name"]    # only one object in one image
    if category in image_item_dict:
        image_item_dict[category].append(item)


def format_bbox(bbox_type, bbox_item):
    bbox_type = "hbb" if bbox_type == "horizontal" else "obb"
    coords = json.loads(bbox_item[bbox_type])
    if bbox_type == "hbb":
        return f"({coords[0]}, {coords[1]}),({coords[2]},{coords[3]})"
    else:
        return ",".join(f"({coords[i]},{coords[i+1]})" for i in range(0, 8, 2))


num_images = [2, 3, 4]
cog_qa = []
stat_txt = []

for category in image_item_dict:
    count = 0
    while len(image_item_dict[category]) >= 4:
        qa = copy.deepcopy(constants.QWEN2_VL_FORMAT)
        selected_item = random.sample(image_item_dict[category], random.choice(num_images))
        images = [image["image_path"] for image in selected_item]
        cog_type = random.choice(["text_with_1st", "text_continuous", "all_images", "text_all_images"])
        query_prefix = "".join([f"Image-{i}: <image>\n" for i in range(1, len(images)+1)])

        if cog_type == "text_with_1st":
            for i in range(1, len(images)+1):
                bbox_type = random.choice(["horizontal", "oriented"])
                query_text = query_prefix + random.choice(cog_query.cog_identify)
                if i != 1:
                    query_text = random.choice(cog_query.cog_identify)
                suffix = f"{i}st" if i == 1 else f"{i}nd" if i == 2 else f"{i}th"
                qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
                qa_item[0]["content"] = query_text + " " + random.choice(
                    cog_query.cog_bbox
                ).replace("bounding box", f"{bbox_type} bounding box").replace(
                    "in the image", f"in the {suffix} image"
                )
                bbox = format_bbox(bbox_type, selected_item[i-1]["objects"][0])
                qa_item[1]["content"] = f"<|object_ref_start|>{category}<|object_ref_end|><|box_start|>{bbox}<|box_end|>"
                qa["messages"].extend(qa_item)
        elif cog_type == "text_continuous":
            query_text = query_prefix + random.choice(cog_query.cog_identify)
            qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
            qa_item[0]["content"] = query_text
            qa_item[1]["content"] = category
            qa["messages"].extend(qa_item)
            for i in range(1, len(images)+1):
                bbox_type = random.choice(["horizontal", "oriented"])
                suffix = f"{i}st" if i == 1 else f"{i}nd" if i == 2 else f"{i}th"
                bbox = format_bbox(bbox_type, selected_item[i-1]["objects"][0])
                qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
                qa_item[0]["content"] = (
                    random.choice(cog_query.cog_bbox)
                    .replace("bounding box", f"{bbox_type} bounding box")
                    .replace("in the image", f"in the {suffix} image")
                )
                qa_item[1]["content"] = f"<|object_ref_start|>{category}<|object_ref_end|><|box_start|>{bbox}<|box_end|>"
                qa["messages"].extend(qa_item)
        elif cog_type == "all_images":
            bbox_type = random.choice(["horizontal", "oriented"])
            query_text = query_prefix + random.choice(cog_query.cog_identify) + " "
            query_text += (
                random.choice(cog_query.cog_bbox)
                .replace("bounding box", f"{bbox_type} bounding box")
                .replace("in the image", "in all images")
            )
            qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
            qa_item[0]["content"] = query_text
            qa_item[1]["content"] = ''.join(
                f"Image-{i+1}: <|object_ref_start|>{category}<|object_ref_end|><|box_start|>{format_bbox(bbox_type, item['objects'][0])}<|box_end|>"
                for i, item in enumerate(selected_item)
            )
            qa["messages"].extend(qa_item)
        else:   # text_all_images
            qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
            query_text = query_prefix + random.choice(cog_query.cog_identify)
            qa_item[0]["content"] = query_text
            qa_item[1]["content"] = category
            qa["messages"].extend(qa_item)
            qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
            bbox_type = random.choice(["horizontal", "oriented"])
            qa_item[0]["content"] = (
                random.choice(cog_query.cog_bbox)
                .replace("bounding box", f"{bbox_type} bounding box")
                .replace("in the image", "in all images")
            )
            qa_item[1]["content"] = ''.join(
                f"Image-{i+1}: <|object_ref_start|>{category}<|object_ref_end|><|box_start|>{format_bbox(bbox_type, item['objects'][0])}<|box_end|>"
                for i, item in enumerate(selected_item)
            )
            qa["messages"].extend(qa_item)

        qa["images"] = images
        cog_qa.append(qa)
        count += 1
        for used_item in selected_item:
            image_item_dict[category].remove(used_item)
    stat_txt.append(f"Category: {category:<30} Count: {count:>6}")

print(f"Total samples: {len(cog_qa)}")
random.shuffle(cog_qa)
with open(cog_save_json, 'w') as f:
    json.dump(cog_qa, f, indent=4)
with open(cog_save_txt, 'w') as f:
    for line in stat_txt:
        f.write(line + '\n')
