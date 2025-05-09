import os
import json
import random
import argparse
from tqdm import tqdm
from PIL import Image
from shapely.geometry import Polygon
from visual_prompt_generator import image_blending


VP_CATEGORY = [
    "rectangle",
    "ellipse",
    "triangle",
    "point",
    "scribble",
    # "mask_contour",
    # "mask",
    "arrow",
]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        "-n",
        type=str,
        default="DOTA-v2_0",
        help="Name of the dataset",
    )
    parser.add_argument(
        "--suffix",
        "-s",
        type=str,
        default="1_to_4",
        help="suffix of the dataset",
    )
    args = parser.parse_args()

    json_path = f"/home/anxiao/Datasets/MIGRANT/DIOR-R/label_{args.suffix}.json"
    output_json = f"/home/anxiao/Datasets/MIGRANT/DIOR-R/vp_{args.suffix}_obj.json"
    output_image_dir = f"/home/anxiao/Datasets/MIGRANT/DIOR-R/vp_{args.suffix}_obj"
    os.makedirs(output_image_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    output_json_data = []

    for item in tqdm(data):
        image = Image.open(item["image_path"])
        if len(item["objects"]) == 1:
            obj = item["objects"][0]  # only one object per image
        else:
            obj = random.choice(item["objects"])  # more than one object per image
        vp = random.choice(VP_CATEGORY)
        if vp == "rectangle":
            bbox = random.choice(["hbb", "obb"])
        else:
            bbox = "obb"

        bndbox = json.loads(obj[bbox])
        if len(bndbox) == 4:
            polygon = Polygon(
                [
                    (bndbox[0], bndbox[1]),
                    (bndbox[2], bndbox[1]),
                    (bndbox[2], bndbox[3]),
                    (bndbox[0], bndbox[3]),
                ]
            )
        else:
            polygon = Polygon(
                [
                    (bndbox[0], bndbox[1]),
                    (bndbox[2], bndbox[3]),
                    (bndbox[4], bndbox[5]),
                    (bndbox[6], bndbox[7]),
                ]
            )
        area = polygon.area
        if area < 400:
            continue

        color, img_drawn = image_blending(
            image=image,
            shape=vp,
            bbox_coord=bndbox,
        )
        img_drawn.save(
            os.path.join(output_image_dir, f"{item['image_path'].split('/')[-1]}")
        )

        output_json_item = {
            "image_path_ori": item["image_path"],
            "image_path": os.path.join(
                output_image_dir, f"{item['image_path'].split('/')[-1]}"
            ),
            "objects": [obj],
            "vp": vp,
            "color": color,
        }

        output_json_data.append(output_json_item)

    with open(output_json, "w") as f:
        json.dump(output_json_data, f, indent=4)
