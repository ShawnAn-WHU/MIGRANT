import os
import copy
import json
import random
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import constants, cvg_query


cvg_json = "/home/anxiao/Datasets/MIGRANT/VIGOR/cross_view_grounding.json"
cvg_save_json = "/home/anxiao/Datasets/MIGRANT/sft/cvg.json"
os.makedirs(os.path.dirname(cvg_save_json), exist_ok=True)

with open(cvg_json, "r") as f:
    cvg_data = json.load(f)
random.shuffle(cvg_data)
cvg_data_ref = cvg_data


def format_point(point, sate_path):
    image = Image.open(sate_path)
    width, height = image.size
    return f"({int(point[0] / width * 1000)},{int(point[1] / height * 1000)})"


num_images = [2, 3, 4]
cvg_qa = []

for item in tqdm(cvg_data[:12000]):
    pano_path = item["pano_path"]
    sate_path = item["sate_path"]
    sate_plot_path = item["sate_plot_path"]
    point = item["point_coords"]

    pano_prefix = "Panoramic Image: <image>\n"
    sate_prefix = "Satellite Image: <image>\n"
    query_prefix = (
        "These are a series of <type1> image followed by several <type2> images. "
    )
    cross_type = random.choice(
        ["match_pano_sate", "pano2sate", "sate2pano", "sate2pano_vp", "sate2pano_tp"]
    )
    qa = copy.deepcopy(constants.QWEN2_VL_FORMAT)

    if len(cvg_data_ref) < 4:
        break

    if cross_type == "match_pano_sate":
        qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
        image_prefix = random.choice(
            [pano_prefix + sate_prefix, sate_prefix + pano_prefix]
        )
        if image_prefix == pano_prefix + sate_prefix:
            qa["images"] = [pano_path, sate_path]
        else:
            qa["images"] = [sate_path, pano_path]
        qa_item[0]["content"] = image_prefix + random.choice(cvg_query.cvg_match)
        qa_item[1]["content"] = f"<|point|>{format_point(point, sate_path)}<|point|>"
        qa["messages"].extend(qa_item)
    elif cross_type == "pano2sate":
        selected_items = random.sample(cvg_data_ref, random.choice(num_images))
        selected_images = [item["sate_path"] for item in selected_items]
        sate_images_list = [sate_path] + selected_images
        random.shuffle(sate_images_list)
        images_list = [pano_path] + sate_images_list
        sate_gt_index = images_list.index(sate_path) + 1
        qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
        qa["images"] = images_list
        image_prefix = "".join(
            f"Image-{i+1}: <image>\n" for i in range(len(images_list))
        )
        query_prefix = query_prefix.replace("<type1>", "panoramic").replace(
            "<type2>", "satellite"
        )
        if random.choice(["separate", "continuous"]) == "separate":
            qa_item[0]["content"] = (
                image_prefix + query_prefix + random.choice(cvg_query.cvg_pano2sate)
            )
            qa_item[1]["content"] = f"Image-{sate_gt_index}."
            qa["messages"].extend(qa_item)
            qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
            qa_item[0]["content"] = random.choice(cvg_query.cvg_point)
            qa_item[1][
                "content"
            ] = f"<|point|>{format_point(point, sate_path)}<|point|>"
            qa["messages"].extend(qa_item)
        else:
            qa_item[0]["content"] = (
                image_prefix
                + random.choice(cvg_query.cvg_pano2sate)
                + random.choice(cvg_query.cvg_point)
            )
            qa_item[1][
                "content"
            ] = f"Image-{sate_gt_index}: <|point|>{format_point(point, sate_path)}<|point|>"
            qa["messages"].extend(qa_item)
        for used_item in selected_items:
            cvg_data_ref.remove(used_item)
    elif cross_type == "sate2pano":
        selected_items = random.sample(cvg_data_ref, random.choice(num_images))
        selected_images = [item["pano_path"] for item in selected_items]
        pano_images_list = [pano_path] + selected_images
        random.shuffle(pano_images_list)
        images_list = [sate_path] + pano_images_list
        pano_gt_index = images_list.index(pano_path) + 1
        qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
        qa["images"] = images_list
        image_prefix = "".join(
            f"Image-{i+1}: <image>\n" for i in range(len(images_list))
        )
        query_prefix = query_prefix.replace("<type1>", "satellite").replace(
            "<type2>", "panoramic"
        )
        if random.choice(["separate", "continuous"]) == "separate":
            qa_item[0]["content"] = (
                image_prefix + query_prefix + random.choice(cvg_query.cvg_sate2pano)
            )
            qa_item[1]["content"] = f"Image-{pano_gt_index}."
            qa["messages"].extend(qa_item)
            qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
            qa_item[0]["content"] = random.choice(cvg_query.cvg_point)
            qa_item[1][
                "content"
            ] = f"<|point|>{format_point(point, sate_path)}<|point|>"
            qa["messages"].extend(qa_item)
        else:
            qa_item[0]["content"] = (
                image_prefix
                + query_prefix
                + random.choice(cvg_query.cvg_sate2pano)
                + " "
                + random.choice(cvg_query.cvg_point)
            )
            qa_item[1][
                "content"
            ] = f"Image-{pano_gt_index}: <|point|>{format_point(point, sate_path)}<|point|>"
            qa["messages"].extend(qa_item)
        for used_item in selected_items:
            cvg_data_ref.remove(used_item)
    elif cross_type == "sate2pano_vp":
        selected_items = random.sample(cvg_data_ref, random.choice(num_images))
        selected_images = [item["pano_path"] for item in selected_items]
        pano_images_list = [pano_path] + selected_images
        random.shuffle(pano_images_list)
        images_list = [sate_plot_path] + pano_images_list
        pano_gt_index = images_list.index(pano_path) + 1
        qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
        qa["images"] = images_list
        image_prefix = "".join(
            f"Image-{i+1}: <image>\n" for i in range(len(images_list))
        )
        query_prefix = query_prefix.replace("<type1>", "satellite").replace(
            "<type2>", "panoramic"
        )
        qa_item[0]["content"] = (
            image_prefix
            + query_prefix
            + random.choice(cvg_query.cvg_sate2pano).replace(
                "the satellite", "the place marked by a red point in the satellite"
            )
        )
        qa_item[1]["content"] = f"Image-{pano_gt_index}."
        qa["messages"].extend(qa_item)
        for used_item in selected_items:
            cvg_data_ref.remove(used_item)
    else:  # sate2pano_tp
        selected_items = random.sample(cvg_data_ref, random.choice(num_images))
        selected_images = [item["pano_path"] for item in selected_items]
        pano_images_list = [pano_path] + selected_images
        random.shuffle(pano_images_list)
        images_list = [sate_path] + pano_images_list
        pano_gt_index = images_list.index(pano_path) + 1
        qa_item = copy.deepcopy(constants.QWEN2_VL_QA_FORMAT)
        qa["images"] = images_list
        image_prefix = "".join(
            f"Image-{i+1}: <image>\n" for i in range(len(images_list))
        )
        query_prefix = query_prefix.replace("<type1>", "satellite").replace(
            "<type2>", "panoramic"
        )
        qa_item[0]["content"] = (
            image_prefix
            + query_prefix
            + random.choice(cvg_query.cvg_sate2pano).replace(
                "the satellite",
                f"the place <|point|>{format_point(point, sate_path)}<|point|> in the satellite",
            )
        )
        qa_item[1]["content"] = f"Image-{pano_gt_index}."
        qa["messages"].extend(qa_item)
        for used_item in selected_items:
            cvg_data_ref.remove(used_item)

    cvg_qa.append(qa)

if len(cvg_data_ref) < 4:
    print("Not enough data for further processing.")
print(f"Total samples: {len(cvg_qa)}")
with open(cvg_save_json, "w") as f:
    json.dump(cvg_qa, f, indent=4)
