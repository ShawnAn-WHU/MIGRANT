import os
import copy
import json
import random
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import constants, icg_query


DOTA_v2_0_region = "/home/anxiao/Datasets/MIGRANT/DOTA-v2_0/region_3_to_10000_obj.json"
DIOR_R_region = "/home/anxiao/Datasets/MIGRANT/DIOR-R/region_3_to_10000_obj.json"
icg_save_json = "/home/anxiao/Datasets/MIGRANT/sft/icg.json"
icg_save_txt = "/home/anxiao/Datasets/MIGRANT/stat_txt/icg.txt"
os.makedirs(os.path.dirname(icg_save_json), exist_ok=True)
os.makedirs(os.path.dirname(icg_save_txt), exist_ok=True)

with open(DOTA_v2_0_region, "r") as f:
    DOTA_v2_0_region = json.load(f)
with open(DIOR_R_region, "r") as f:
    DIOR_R_region = json.load(f)
icg_data = DOTA_v2_0_region + DIOR_R_region


def format_bbox(bbox_type, bbox_item, image_path_src):
    image = Image.open(image_path_src)
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

icg_qa = []
for item in tqdm(icg_data):
    image_path_src = item["image_path"]
    num_regions = len(item["objects"])
    region_paths = [item[f"region_{i+1}"] for i in range(num_regions)]
    bbox_type = random.choice(["horizontal", "oriented"])
    image_prefix = "Source Image: <image>\n" + "".join(
        [f"Region-{i+1}: <image>\n" for i in range(num_regions)]
    )
    query_prefix = "These are a series of source image followed by its region crops. "
    icg_type = random.choice(
        ["identify_with_1st", "identify_continuous", "all_regions"]
    )
    qa = copy.deepcopy(constants.QWEN2_VL_FORMAT)

    if icg_type == "identify_with_1st":
        for i in range(1, num_regions + 1):
            query_identify = random.choice(icg_query.icg_identify).replace(
                "Region-", f"Region-{i}"
            )
            query_text = image_prefix + query_prefix + query_identify
            if i != 1:
                query_text = query_identify
            suffix = f"{i}st" if i == 1 else f"{i}nd" if i == 2 else f"{i}th"
            qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
            qa_item[0]["content"] = (
                query_text
                + " "
                + random.choice(icg_query.icg_bbox)
                .replace("bounding box", f"{bbox_type} bounding box")
                .replace("in the image", "in the source image")
            )
            bbox = format_bbox(bbox_type, item["objects"][i - 1], image_path_src)
            qa_item[1][
                "content"
            ] = f"<|object_ref_start|>{item['objects'][i - 1]['object_name']}<|object_ref_end|><|box_start|>{bbox}<|box_end|>"
            qa["messages"].extend(qa_item)
    elif icg_type == "identify_continuous":
        for i in range(1, num_regions + 1):
            query_identify = random.choice(icg_query.icg_identify).replace(
                "Region-", f"Region-{i}"
            )
            query_text = image_prefix + query_prefix + query_identify
            if i != 1:
                query_text = query_identify
            qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
            qa_item[0]["content"] = query_text
            qa_item[1]["content"] = f"{item['objects'][i-1]['object_name']}"
            qa["messages"].extend(qa_item)
            qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
            qa_item[0]["content"] = (
                random.choice(icg_query.icg_bbox)
                .replace("bounding box", f"{bbox_type} bounding box")
                .replace("in the image", "in the source image")
            )
            bbox = format_bbox(bbox_type, item["objects"][i - 1], image_path_src)
            qa_item[1][
                "content"
            ] = f"<|object_ref_start|>{item['objects'][i - 1]['object_name']}<|object_ref_end|><|box_start|>{bbox}<|box_end|>"
            qa["messages"].extend(qa_item)
    elif icg_type == "all_regions":
        query_text = (
            image_prefix
            + query_prefix
            + random.choice(icg_query.icg_identify).replace("Region-", "each region")
            + " "
        )
        query_text += random.choice(icg_query.icg_bbox_all_region).replace(
            "bounding box", f"{bbox_type} bounding box"
        )
        qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
        qa_item[0]["content"] = query_text
        qa_item[1]["content"] = "".join(
            f"Region-{i+1}: <|object_ref_start|>{item['objects'][i]['object_name']}<|object_ref_end|><|box_start|>{format_bbox(bbox_type, item['objects'][i], image_path_src)}<|box_end|>"
            for i in range(num_regions)
        )
        qa["messages"].extend(qa_item)

    qa["images"] = [image_path_src] + region_paths
    icg_qa.append(qa)

print(f"Total samples: {len(icg_qa)}")
with open(icg_save_json, "w") as f:
    json.dump(icg_qa, f, indent=4)
