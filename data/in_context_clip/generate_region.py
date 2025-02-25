import os
import json
import random
from PIL import Image
from tqdm import tqdm
from shapely.geometry import Polygon


def is_area_large(obb, threshold=1000):
    bndbox = json.loads(obb)
    polygon = Polygon(
        [
            (bndbox[0], bndbox[1]),
            (bndbox[2], bndbox[3]),
            (bndbox[4], bndbox[5]),
            (bndbox[6], bndbox[7]),
        ]
    )
    area = polygon.area
    if area < threshold:
        return False
    return True


def clip_region(objects_list, image_path, output_image_dir):
    region_path_list = []
    image = Image.open(image_path)
    width, height = image.size
    for i, obj in enumerate(objects_list):
        bndbox = json.loads(obj["hbb"])
        x_min, y_min, x_max, y_max = bndbox[0], bndbox[1], bndbox[2], bndbox[3]
        if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
            continue
        region_image = image.crop(bndbox)
        region_path = os.path.join(
            output_image_dir, f"{image_path.split('/')[-1].split('.')[0]}_{i+1}.png"
        )
        region_image.save(region_path)
        region_path_list.append(region_path)
    return region_path_list


if __name__ == "__main__":

    json_path = "/home/anxiao/Datasets/MIGRANT/DOTA-v2_0/label_3_to_10000.json"
    output_json = "/home/anxiao/Datasets/MIGRANT/DOTA-v2_0/region_3_to_10000_obj.json"
    output_image_dir = "/home/anxiao/Datasets/MIGRANT/DOTA-v2_0/region_3_to_10000_obj"
    os.makedirs(output_image_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    output_json_data = []

    for item in tqdm(data):
        image_path = item["image_path"]
        objects_list = []
        for obj in item["objects"]:
            if is_area_large(obj["obb"], 1000):
                objects_list.append(obj)
        if len(objects_list) < 3:
            continue

        large_objects_list = random.sample(objects_list, random.randint(3, len(objects_list)))

        region_path_list = clip_region(large_objects_list, image_path, output_image_dir)
        if region_path_list is None or len(region_path_list) < 3:
            continue
        output_json_item = {
            "image_path": item["image_path"],
        }
        for i in range(len(region_path_list)):
            output_json_item[f"region_{i}"] = region_path_list[i]
        for i in range(len(region_path_list)):
            output_json_item["objects"] = objects_list

        output_json_data.append(output_json_item)

    with open(output_json, "w") as f:
        json.dump(output_json_data, f, indent=4)
