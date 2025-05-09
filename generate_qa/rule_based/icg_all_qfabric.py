import os
import copy
import json
import random
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import constants, icg_query


QFabric_region = "/home/anxiao/Datasets/MIGRANT/QFabric/region_all.json"
icg_save_json = "/home/anxiao/Datasets/MIGRANT/sft/icg_qfabric.json"
os.makedirs(os.path.dirname(icg_save_json), exist_ok=True)

with open(QFabric_region, "r") as f:
    QFabric_region = json.load(f)


def format_bbox(bbox_type, bbox_item, image_path_src):
    image = Image.open(image_path_src)
    width, height = image.size
    bbox_type = "hbb" if bbox_type == "horizontal" else "obb"
    coords = bbox_item[bbox_type]
    if bbox_type == "hbb":
        return f"({int(coords[0] / width * 1000)},{int(coords[1] / height * 1000)}),({int(coords[2] / width * 1000)},{int(coords[3] / height * 1000)})"
    else:
        return ",".join(
            f"({int(coords[i] / width * 1000)},{int(coords[i+1] / height * 1000)})"
            for i in range(0, 8, 2)
        )


def split_chunks(region_paths, max_size=4):
    chunks = []
    i = 0
    while i < len(region_paths):
        remaining = len(region_paths) - i
        chunk_size = (
            random.randint(2, min(max_size, remaining) + 1) if remaining > 1 else 1
        )
        chunks.append(region_paths[i : i + chunk_size])
        i += chunk_size

    return chunks


icg_qa = []
for item in tqdm(QFabric_region):
    image_path_src = item["image_path"]
    num_regions = len(item["objects"])
    region_paths = [item[f"region_{i+1}"] for i in range(num_regions)]
    chunks = split_chunks(region_paths)
    index = 0
    for chunk in chunks:
        num_regions = len(chunk)
        bbox_type = "horizontal"
        image_prefix = "Source Image: <image>\n" + "".join(
            [f"Region-{i+1}: <image>\n" for i in range(num_regions)]
        )
        query_prefix = (
            "These are a series of source image followed by its region crops. "
        )
        icg_type = random.choice(["locate_with_1st", "all_regions"])
        qa = copy.deepcopy(constants.QWEN2_VL_FORMAT)

        if icg_type == "locate_with_1st":
            for i in range(1, num_regions + 1):
                query_locate = random.choice(icg_query.icg_locate).replace(
                    "Region-1", f"Region-{i}"
                )
                query_text = image_prefix + query_prefix + query_locate
                if i != 1:
                    query_text = query_locate
                suffix = f"{i}st" if i == 1 else f"{i}nd" if i == 2 else f"{i}th"
                qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
                qa_item[0]["content"] = query_text
                bbox = format_bbox(
                    bbox_type, item["objects"][i + index - 1], image_path_src
                )
                qa_item[1]["content"] = f"<|box_start|>{bbox}<|box_end|>"
                qa["messages"].extend(qa_item)
        elif icg_type == "all_regions":
            query_text = (
                image_prefix
                + query_prefix
                + random.choice(icg_query.icg_locate).replace("Region-1", "each region")
            )
            qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
            qa_item[0]["content"] = query_text
            qa_item[1]["content"] = "".join(
                f"Region-{i+1}: <|box_start|>{format_bbox(bbox_type, item['objects'][i], image_path_src)}<|box_end|>\n"
                for i in range(num_regions)
            )
            qa["messages"].extend(qa_item)

        index += len(chunk)
        qa["images"] = [image_path_src] + chunk
        icg_qa.append(qa)

print(f"Total samples: {len(icg_qa)}")
with open(icg_save_json, "w") as f:
    json.dump(icg_qa, f, indent=4)
