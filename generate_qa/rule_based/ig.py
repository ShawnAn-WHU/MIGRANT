import os
import copy
import json
import random
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import constants, ig_query


DOTA_v2_0_vp = "/home/anxiao/Datasets/MIGRANT/DOTA-v2_0/vp_1_obj.json"
DIOR_R_vp = "/home/anxiao/Datasets/MIGRANT/DIOR-R/vp_1_obj.json"
DOTA_v2_0_2_to_10 = "/home/anxiao/Datasets/MIGRANT/DOTA-v2_0/label_2_to_10.json"
DIOR_R_2_to_10 = "/home/anxiao/Datasets/MIGRANT/DIOR-R/label_2_to_10.json"
ig_save_json = "/home/anxiao/Datasets/MIGRANT/sft/ig.json"
ig_save_txt = "/home/anxiao/Datasets/MIGRANT/stat_txt/ig.txt"
os.makedirs(os.path.dirname(ig_save_json), exist_ok=True)
os.makedirs(os.path.dirname(ig_save_txt), exist_ok=True)

with open(DOTA_v2_0_vp, "r") as f:
    DOTA_v2_0_vp = json.load(f)
with open(DIOR_R_vp, "r") as f:
    DIOR_R_vp = json.load(f)
ig_data = DOTA_v2_0_vp + DIOR_R_vp
random.shuffle(ig_data)

with open(DOTA_v2_0_2_to_10, "r") as f:
    DOTA_v2_0_2_to_10 = json.load(f)
with open(DIOR_R_2_to_10, "r") as f:
    DIOR_R_2_to_10 = json.load(f)
ref_data = DOTA_v2_0_2_to_10 + DIOR_R_2_to_10

image_item_dict_ref = {cat: [] for cat in constants.DOTA_DIOR_COMBINE}
for category in image_item_dict_ref:
    for item in ref_data:
        new_item = item.copy()
        new_item["objects"] = []
        for i in range(len(item["objects"])):
            cat = item["objects"][i]["object_name"]
            if cat == category:
                new_item["objects"].append(item["objects"][i])
        if len(new_item["objects"]) == 0:
            continue
        image_item_dict_ref[category].append(new_item)


def format_bbox(bbox_type, bbox_item, item):
    image = Image.open(item["image_path"])
    width, height = image.size
    bbox_type = "hbb" if bbox_type == "horizontal" else "obb"
    coords = json.loads(bbox_item[bbox_type])
    if bbox_type == "hbb":
        return f"({int(coords[0] / width * 1000)},{int(coords[1] / height * 1000)}),({int(coords[2] / width * 1000)},{int(coords[3] / height * 1000)})"
    else:
        return ",".join(
            f"({int(coords[i] / width * 1000)},{int(coords[i+1] / height * 1000)})"
            for i in range(0, 8, 2)
        )

num_images = [2, 3, 4, 5, 6]
ig_qa = []
count = {cat: 0 for cat in constants.DOTA_DIOR_COMBINE}

for item in tqdm(ig_data):
    category = item["objects"][0]["object_name"]
    if len(image_item_dict_ref[category]) < 6:
        # print(f"Category {category} has less than 6 images, skipping this category.")
        continue
    selected_items = random.sample(
        image_item_dict_ref[category], random.choice(num_images)
    )
    images = [item["image_path"]]
    images_selected = [image["image_path"] for image in selected_items]
    images.extend(images_selected)

    vp = item["vp"]
    color = item["color"]
    if vp == "arrow":
        identify_query = random.choice(ig_query.ig_identify_arrow)
    elif vp == "ellipse":
        identify_query = random.choice(ig_query.ig_identify_ellipse)
    elif vp == "rectangle":
        identify_query = random.choice(ig_query.ig_identify_rectangle)
    elif vp == "point":
        identify_query = random.choice(ig_query.ig_identify_point)
    elif vp == "scribble":
        identify_query = random.choice(ig_query.ig_identify_scribble)
    else:
        identify_query = random.choice(ig_query.ig_identify_triangle)

    if "." in identify_query:
        identify_query = identify_query.replace(
            ".",
            random.choice(
                [" in Image-1.", " in the 1st image.", " in the first image."]
            ),
        )
    elif "?" in identify_query:
        identify_query = identify_query.replace(
            "?",
            random.choice(
                [" in Image-1?", " in the 1st image?", " in the first image?"]
            ),
        )
    else:
        raise ValueError("Identify query must end with '.' or '?'")

    qa = copy.deepcopy(constants.QWEN2_VL_FORMAT)
    ig_type = random.choice(
        [
            "identify_with_second",
            "identify_continuous",
            "all_images",
            "identify_all_images",
        ]
    )
    query_prefix = "".join([f"Image-{i}: <image>\n" for i in range(1, len(images) + 1)])

    if ig_type == "identify_with_second":
        for i in range(2, len(images) + 1):
            bbox_type = random.choice(["horizontal", "oriented"])
            query_text = query_prefix + identify_query.replace("<color>", color)
            if i != 2:
                query_text = identify_query.replace("<color>", color)
            suffix = f"{i}nd" if i == 2 else f"{i}th"
            qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
            qa_item[0]["content"] = (
                query_text
                + " "
                + random.choice(ig_query.ig_bbox)
                .replace("bounding box", f"{bbox_type} bounding box")
                .replace("in the image", f"in the {suffix} image")
            )
            qa_item[1]["content"] = "".join(
                f"<|object_ref_start|>{object_item['object_name']}<|object_ref_end|><|box_start|>{format_bbox(bbox_type, object_item, selected_items[i - 2])}<|box_end|>"
                for object_item in selected_items[i - 2]["objects"]
            )
            qa["messages"].extend(qa_item)
    elif ig_type == "identify_continuous":
        query_text = query_prefix + identify_query.replace("<color>", color)
        qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
        qa_item[0]["content"] = query_text
        qa_item[1]["content"] = category
        qa["messages"].extend(qa_item)
        for i in range(2, len(images) + 1):
            bbox_type = random.choice(["horizontal", "oriented"])
            suffix = f"{i}nd" if i == 2 else f"{i}th"
            qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
            qa_item[0]["content"] = (
                random.choice(ig_query.ig_bbox)
                .replace("bounding box", f"{bbox_type} bounding box")
                .replace("in the image", f"in the {suffix} image")
            )
            qa_item[1]["content"] = "".join(
                f"<|object_ref_start|>{object_item['object_name']}<|object_ref_end|><|box_start|>{format_bbox(bbox_type, object_item, selected_items[i - 2])}<|box_end|>"
                for object_item in selected_items[i - 2]["objects"]
            )
            qa["messages"].extend(qa_item)
    elif ig_type == "all_images":
        bbox_type = random.choice(["horizontal", "oriented"])
        query_text = query_prefix + identify_query.replace("<color>", color) + " "
        query_text += (
            random.choice(ig_query.ig_bbox)
            .replace("bounding box", f"{bbox_type} bounding box")
            .replace("in the image", "in all other images")
        )
        qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
        qa_item[0]["content"] = query_text
        for i, item in enumerate(selected_items):
            qa_item[1]["content"] += f"Image-{i+2}: "
            for j in range(len(item["objects"])):
                qa_item[1][
                    "content"
                ] += f"<|object_ref_start|>{item['objects'][j]['object_name']}<|object_ref_end|><|box_start|>{format_bbox(bbox_type, item['objects'][j], item)}<|box_end|>"
            qa_item[1]["content"] += "\n"
        qa["messages"].extend(qa_item)
    else:  # identify_all_images
        qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
        query_text = query_prefix + identify_query.replace("<color>", color)
        qa_item[0]["content"] = query_text
        qa_item[1]["content"] = category
        qa["messages"].extend(qa_item)
        qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
        bbox_type = random.choice(["horizontal", "oriented"])
        qa_item[0]["content"] = (
            random.choice(ig_query.ig_bbox)
            .replace("bounding box", f"{bbox_type} bounding box")
            .replace("in the image", "in all other images")
        )
        for i, item in enumerate(selected_items):
            qa_item[1]["content"] += f"Image-{i+2}: "
            for j in range(len(item["objects"])):
                qa_item[1][
                    "content"
                ] += f"<|object_ref_start|>{item['objects'][j]['object_name']}<|object_ref_end|><|box_start|>{format_bbox(bbox_type, item['objects'][j], item)}<|box_end|>"
            qa_item[1]["content"] += "\n"
        qa["messages"].extend(qa_item)

    qa["images"] = images
    ig_qa.append(qa)
    count[category] += 1
    for used_item in selected_items:
        image_item_dict_ref[category].remove(used_item)

print(f"Total samples: {len(ig_qa)}")
with open(ig_save_json, "w") as f:
    json.dump(ig_qa, f, indent=4)
with open(ig_save_txt, "w") as f:
    for cat in count:
        f.write(f"{cat:<30}: {count[cat]:>6}\n")
