import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
from shapely.geometry import Polygon


def is_area_large(bbox, threshold=1000):
    bndbox = json.loads(bbox)
    if len(bndbox) == 8:
        polygon = Polygon(
            [
                (bndbox[0], bndbox[1]),
                (bndbox[2], bndbox[3]),
                (bndbox[4], bndbox[5]),
                (bndbox[6], bndbox[7]),
            ]
        )
    else:
        polygon = Polygon(
            [
                (bndbox[0], bndbox[1]),
                (bndbox[2], bndbox[1]),
                (bndbox[2], bndbox[3]),
                (bndbox[0], bndbox[3]),
            ]
        )
    area = polygon.area
    if area < threshold:
        return False
    return True


def clip_region(objects_list, image_path, output_image_dir):
    image = Image.open(image_path)
    width, height = image.size
    region_path_list = []
    region_images = {}
    for i, obj in enumerate(objects_list):
        bndbox = json.loads(obj["hbb"])
        x_min, y_min, x_max, y_max = bndbox[0], bndbox[1], bndbox[2], bndbox[3]
        if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
            continue
        if ((x_max - x_min) / (y_max - y_min) > 2) or (
            (y_max - y_min) / (x_max - x_min) > 2
        ):
            continue
        region_image = image.crop(bndbox)
        region_images[f"str{i}"] = [region_image, obj]

    suitable_objects_list = []
    if len(region_images) > 2:
        region_images = list(region_images.values())
        for i, (region_image, obj) in enumerate(region_images):
            region_path = os.path.join(
                output_image_dir, f"{image_path.split('/')[-1].split('.')[0]}_{i+1}.png"
            )
            region_image.save(region_path)
            region_path_list.append(region_path)
            suitable_objects_list.append(obj)
    return region_path_list, suitable_objects_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        type=str,
        default="/home/anxiao/Datasets/MIGRANT/DOTA-v2_0/label_3_to_10000.json",
        help="Path to the input JSON file.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="/home/anxiao/Datasets/MIGRANT/DOTA-v2_0/region_3_to_10000_obj_all.json",
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--output_image_dir",
        type=str,
        default="/home/anxiao/Datasets/MIGRANT/DOTA-v2_0/region_3_to_10000_obj_all",
        help="Directory to save the output images.",
    )
    args = parser.parse_args()

    json_path = args.json_path
    output_json = args.output_json
    output_image_dir = args.output_image_dir
    os.makedirs(output_image_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    output_json_data = []

    for item in tqdm(data):
        image_path = item["image_path"]
        objects_list = []
        for obj in item["objects"]:
            if "obb" in obj:
                if is_area_large(obj["obb"], 10000):
                    objects_list.append(obj)
            else:
                if is_area_large(obj["hbb"], 14400):
                    objects_list.append(obj)
        if len(objects_list) < 2:
            continue

        region_path_list, suitable_objects_list = clip_region(
            objects_list, image_path, output_image_dir
        )
        if not region_path_list and not suitable_objects_list:
            continue
        output_json_item = {
            "image_path": item["image_path"],
        }
        for i in range(len(region_path_list)):
            output_json_item[f"region_{i+1}"] = region_path_list[i]
        output_json_item["objects"] = suitable_objects_list
        output_json_data.append(output_json_item)

    with open(output_json, "w") as f:
        json.dump(output_json_data, f, indent=4)
