import os
import copy
import json
import random
from PIL import Image
from tqdm import tqdm

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import constants, csg_query


satellite_json = "/home/anxiao/Datasets/MIGRANT/CSG/coords_new.json"
map_json = "/home/anxiao/Datasets/MIGRANT/CSG/coords_map_new.json"
csg_save_json = "/home/anxiao/Datasets/MIGRANT/sft/csg.json"
os.makedirs(os.path.dirname(csg_save_json), exist_ok=True)

with open(satellite_json, "r") as f:
    satellite_data = json.load(f)
with open(map_json, "r") as f:
    map_data = json.load(f)


def format_bbox(bbox_type, bbox_item):
    image = Image.open(bbox_item["image_path"])
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


csg_qa = []
for item_sate, item_map in tqdm(zip(satellite_data, map_data)):
    sate_path = item_sate["image_path"]
    map_path = item_map["image_path"]
    sate_plot_path = sate_path.replace("/png/", "/png_plot/")
    map_plot_path = map_path.replace("/png_map/", "/png_map_plot/")
    if not os.path.exists(sate_path) or not os.path.exists(map_path):
        print(f"Image not found: {sate_path}")
        continue
    if not os.path.exists(sate_plot_path) or not os.path.exists(map_plot_path):
        print(f"Image not found: {sate_plot_path}")
        continue
    cross_type = random.choice(["sate2map", "map2sate"])
    sate_prefix = "Satellite Image: <image>\n"
    map_prefix = "Map Image: <image>\n"
    qa = copy.deepcopy(constants.QWEN2_VL_FORMAT)
    qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
    if cross_type == "sate2map":
        prompt_type = random.choice(["vp", "tp"])
        bbox_type = "horizontal" if "hbb" in item_map else "oriented"
        sate_bbox_type = "horizontal" if "hbb" in item_sate else "oriented"
        image_prefix = sate_prefix + map_prefix
        if prompt_type == "vp":
            query_text = (
                random.choice(csg_query.csg_vp_bbox)
                .replace("<image1>", "the Satellite Image")
                .replace("<image2>", "the Map Image")
                .replace("bounding box", f"{bbox_type} bounding box")
            )
            qa["images"] = [sate_plot_path, map_path]
        else:  # tp
            query_text = (
                random.choice(csg_query.csg_tp_bbox)
                .replace("<image1>", "the Satellite Image")
                .replace("<image2>", "the Map Image")
                .replace(
                    "the object",
                    f"the object <|box_start|>{format_bbox(sate_bbox_type, item_sate)}<|box_end|>",
                )
                .replace("bounding box", f"{bbox_type} bounding box")
            )
            qa["images"] = [sate_path, map_path]
        qa_item[0]["content"] = image_prefix + query_text
        qa_item[1][
            "content"
        ] = f"<|box_start|>{format_bbox(bbox_type, item_map)}<|box_end|>"
        qa["messages"].extend(qa_item)
    else:  # map2sate
        prompt_type = random.choice(["vp", "tp"])
        bbox_type = "horizontal" if "hbb" in item_sate else "oriented"
        map_bbox_type = "horizontal" if "hbb" in item_map else "oriented"
        image_prefix = map_prefix + sate_prefix
        if prompt_type == "vp":
            query_text = (
                random.choice(csg_query.csg_vp_bbox)
                .replace("<image1>", "the Map Image")
                .replace("<image2>", "the Satellite Image")
                .replace("bounding box", f"{bbox_type} bounding box")
            )
            qa["images"] = [map_plot_path, sate_path]
        else:  # tp
            query_text = (
                random.choice(csg_query.csg_tp_bbox)
                .replace("<image1>", "the Map Image")
                .replace("<image2>", "the Satellite Image")
                .replace(
                    "the object",
                    f"the object <|box_start|>{format_bbox(map_bbox_type, item_map)}<|box_end|>",
                )
                .replace("bounding box", f"{bbox_type} bounding box")
            )
            qa["images"] = [map_path, sate_path]
        qa_item[0]["content"] = image_prefix + query_text
        qa_item[1][
            "content"
        ] = f"<|box_start|>{format_bbox(bbox_type, item_sate)}<|box_end|>"
        qa["messages"].extend(qa_item)
    csg_qa.append(qa)

print(f"Total samples: {len(csg_qa)}")
with open(csg_save_json, "w") as f:
    json.dump(csg_qa, f, indent=4)
