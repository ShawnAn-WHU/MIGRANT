import os
import re
import json
import math
import copy
import random
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import constants


def hbb_or_obb(coords):

    def angle(p1, p2):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        return math.degrees(math.atan2(dy, dx))

    angles = [angle(coords[i], coords[(i + 1) % 4]) % 180 for i in range(4)]
    is_hbb = all(abs(a - 0) < 5 or abs(a - 90) < 5 for a in angles)

    if is_hbb:
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        return [x1, y1, x2, y2]
    else:
        return [coord for point in coords for coord in point]


def rotate_bbox(bbox, angle):
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2
    angle_rad = np.deg2rad(angle)
    rotation_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )
    vertices = np.array(
        [
            [bbox[0], bbox[1]],  # top-left
            [bbox[2], bbox[1]],  # top-right
            [bbox[2], bbox[3]],  # bottom-right
            [bbox[0], bbox[3]],  # bottom-left
        ]
    )

    vertices_shifted = vertices - [x_center, y_center]
    rotated_vertices = vertices_shifted.dot(rotation_matrix.T)
    rotated_vertices += [x_center, y_center]
    rotated_vertices = np.clip(rotated_vertices, 0, 1000)
    rotated_vertices = rotated_vertices.squeeze().astype(int).tolist()
    return hbb_or_obb(rotated_vertices)


geochat_image_root = "/home/anxiao/Datasets/GeoChat_Instruct/images"
geochat_instruct_json = "/home/anxiao/Datasets/GeoChat_Instruct/GeoChat_Instruct.json"
save_qwen_format_json = "/home/anxiao/Datasets/GeoChat_Instruct/GeoChat_Instruct_qwen.json"

with open(geochat_instruct_json, "r") as f:
    data = json.load(f)

geochat_qa = []
for item in tqdm(data):
    image_path = os.path.join(geochat_image_root, item["image"])
    if not os.path.exists(image_path):
        continue

    qa = copy.deepcopy(constants.QWEN2_VL_FORMAT)
    for i in range(0, len(item["conversations"]), 2):
        qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
        query = item["conversations"][i]["value"]
        answer = item["conversations"][i + 1]["value"]
        if "[identify]" in query:
            matches = re.findall(r"{(.*?)}", query)
            if len(matches) > 1:
                raise ValueError(f"Multiple matches found for [identify]: {query}")
            coords = re.findall(r"<(\d+)>", query)
            assert len(coords) == 5, f"Invalid query format: {query}"
            if len(set(coords)) < len(coords):
                continue
            coords = [int(coord) * 10 for coord in coords[:-1]]  # [0:100] -> [0:1000]
            angle = int(coords[-1])
            coords = rotate_bbox(coords, angle)
            if len(coords) == 4:
                coords_text = f"<|box_start|>({coords[0]},{coords[1]}),({coords[2]},{coords[3]})<|box_end|>"
            else:
                coords_text = f"<|box_start|>({coords[0]},{coords[1]}),({coords[2]},{coords[3]}),({coords[4]},{coords[5]}),({coords[6]},{coords[7]})<|box_end|>"
            query = query.replace("[identify]", "identify")
            qa_item[0]["content"] = re.sub(r"{.*?}", f"{coords_text}", query)
            qa_item[1]["content"] = answer.replace(
                "<p>", "<|object_ref_start|>"
            ).replace("</p>", "<|object_ref_end|>")
        elif "refer" in query:
            if "[refer] <p>" in query:
                qa_item[0]["content"] = (
                    query.replace("refer", "Locate")
                    .replace("<p>", "<|object_ref_start|>")
                    .replace("</p>", "<|object_ref_end|>")
                )
            else:
                qa_item[0]["content"] = (
                    query.replace("[refer]", "")
                    .replace("<p>", "<|object_ref_start|>")
                    .replace("</p>", "<|object_ref_end|>")
                )
            all_coords = []
            matches = re.findall(r"{(.*?)}", answer)
            for match in matches:
                coords = re.findall(r"<(\d+)>", match)
                assert len(coords) == 5, f"Invalid answer format: {answer}"
                if len(set(coords)) < len(coords):
                    all_coords = []
                    break
                coords = [int(coord) * 10 for coord in coords[:4]]
                angle = int(coords[-1])
                coords = rotate_bbox(coords, angle)
                if len(coords) == 4:
                    coords_text = f"<|box_start|>({coords[0]},{coords[1]}),({coords[2]},{coords[3]})<|box_end|>"
                else:
                    coords_text = f"<|box_start|>({coords[0]},{coords[1]}),({coords[2]},{coords[3]}),({coords[4]},{coords[5]}),({coords[6]},{coords[7]})<|box_end|>"
                all_coords.append(coords_text)
            if len(all_coords) == 0:
                continue
            qa_item[1]["content"] = "".join(all_coords)
        elif "[grounding]" in query:
            continue
        else:
            qa_item[0]["content"] = query
            qa_item[1]["content"] = answer
        qa["messages"].extend(qa_item)
    qa["images"] = [image_path]
    if len(qa["messages"]) == 1:
        continue
    geochat_qa.append(qa)

random.shuffle(geochat_qa)
print(f"Total samples: {len(geochat_qa)}")
with open(save_qwen_format_json, "w") as f:
    json.dump(geochat_qa, f, indent=4)
