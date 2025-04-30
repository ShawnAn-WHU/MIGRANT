import os
import copy
import json
import random
from PIL import Image
from tqdm import tqdm
from collections import Counter

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import constants, dg_query


DOTA_v2_0_remove_obj = (
    "/home/anxiao/Datasets/MIGRANT/DOTA-v2_0/label_remove_2_to_10.json"
)
DIOR_R_remove_obj = "/home/anxiao/Datasets/MIGRANT/DIOR-R/label_remove_2_to_10.json"
DIOR_R_remove_vehicle = (
    "/home/anxiao/Datasets/MIGRANT/DIOR-R/label_remove_2_to_4_vehicle.json"
)
NWPU_VHR_remove_obj = (
    "/home/anxiao/Datasets/MIGRANT/NWPU-VHR-10/label_remove_2_to_10.json"
)
RSOD_remove_obj = "/home/anxiao/Datasets/MIGRANT/RSOD/label_remove_2_to_10.json"

dg_save_json = "/home/anxiao/Datasets/MIGRANT/sft/dg.json"
os.makedirs(os.path.dirname(dg_save_json), exist_ok=True)
# dg_save_txt = "/home/anxiao/Datasets/MIGRANT/stat_txt/dg.txt"
# os.makedirs(os.path.dirname(dg_save_txt), exist_ok=True)

with open(DOTA_v2_0_remove_obj, "r") as f:
    DOTA_v2_0_remove_obj = json.load(f)
with open(DIOR_R_remove_obj, "r") as f:
    DIOR_R_remove_obj = json.load(f)
with open(DIOR_R_remove_vehicle, "r") as f:
    DIOR_R_remove_vehicle = json.load(f)
with open(NWPU_VHR_remove_obj, "r") as f:
    NWPU_VHR_remove_obj = json.load(f)
with open(RSOD_remove_obj, "r") as f:
    RSOD_remove_obj = json.load(f)

dg_data = (
    DOTA_v2_0_remove_obj
    + DIOR_R_remove_obj
    + DIOR_R_remove_vehicle
    + NWPU_VHR_remove_obj
    + RSOD_remove_obj
)


def format_bbox(bbox_type, bbox_item, image_path_ori):
    image = Image.open(image_path_ori)
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


dg_qa = []
for item in tqdm(dg_data):
    image_path_ori = item["image_path"]
    image_path_remove = item["output_image_path"]
    objects = item["objects"]

    if "obb" in objects[0]:
        bbox_type = random.choice(["horizontal", "oriented"])
    else:
        bbox_type = "horizontal"

    compare = random.choice(["appear", "disappear"])
    dg_type = random.choice(["compare_dg", "describe_dg"])
    query_prefix = "Image-1: <image>\nImage-2: <image>\n"

    qa = copy.deepcopy(constants.QWEN2_VL_FORMAT)
    if dg_type == "compare_dg":
        qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
        qa_item[0]["content"] = (
            query_prefix
            + random.choice(dg_query.dg_compare)
            + " "
            + random.choice(dg_query.dg_bbox).replace(
                "bounding box", f"{bbox_type} bounding box"
            )
        )
        qa_item[1]["content"] = "".join(
            f"<|object_ref_start|>{object_item['object_name']}<|object_ref_end|><|box_start|>{format_bbox(bbox_type, object_item, image_path_ori)}<|box_end|>"
            for object_item in objects
        )
        qa["messages"].extend(qa_item)
    else:
        qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
        qa_item[0]["content"] = query_prefix + random.choice(dg_query.dg_describe)
        object_counts = Counter(object_item["object_name"] for object_item in objects)
        if compare == "appear":
            qa_item[1]["content"] = random.choice(
                dg_query.dg_describe_answer_appear
            ).replace(
                "<object>",
                ", ".join(
                    f"{count} more {name}" for name, count in object_counts.items()
                ),
            )
        else:
            qa_item[1]["content"] = random.choice(
                dg_query.dg_describe_answer_disappear
            ).replace(
                "<object>",
                ", ".join(f"{count} {name}" for name, count in object_counts.items()),
            )
        qa["messages"].extend(qa_item)
        qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
        qa_item[0]["content"] = random.choice(dg_query.dg_bbox).replace(
            "bounding box", f"{bbox_type} bounding box"
        )
        qa_item[1]["content"] = "".join(
            f"<|object_ref_start|>{object_item['object_name']}<|object_ref_end|><|box_start|>{format_bbox(bbox_type, object_item, image_path_ori)}<|box_end|>"
            for object_item in objects
        )
        qa["messages"].extend(qa_item)

    if compare == "appear":
        qa["images"] = [image_path_remove, image_path_ori]
    else:
        qa["images"] = [image_path_ori, image_path_remove]

    dg_qa.append(qa)

hallucination_data = random.sample(dg_data, int(len(dg_data) * 0.3))
for item in tqdm(hallucination_data):
    qa = copy.deepcopy(constants.QWEN2_VL_FORMAT)
    qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
    qa_item[0]["content"] = query_prefix + random.choice(dg_query.dg_describe)
    qa_item[1]["content"] = random.choice(dg_query.dg_hallucination)
    qa["messages"].extend(qa_item)
    qa["images"] = [random.choice([item["image_path"], item["output_image_path"]])] * 2
    dg_qa.append(qa)

random.shuffle(dg_qa)
print(f"Total samples: {len(dg_qa)}")
with open(dg_save_json, "w") as f:
    json.dump(dg_qa, f, indent=4)
