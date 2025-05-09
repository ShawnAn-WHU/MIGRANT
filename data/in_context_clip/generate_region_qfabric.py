import os
import json
from PIL import Image
from tqdm import tqdm
from shapely.geometry import Polygon


def is_area_large(bbox, threshold=1000):
    bndbox = bbox
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
        bndbox = obj["hbb"]
        x_min, y_min, x_max, y_max = bndbox[0], bndbox[1], bndbox[2], bndbox[3]
        if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
            continue
        if ((x_max - x_min) / (y_max - y_min) > 3) or (
            (y_max - y_min) / (x_max - x_min) > 3
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

    json_path = "/home/anxiao/Datasets/MIGRANT/QFabric/icg/label.json"
    output_json = "/home/anxiao/Datasets/MIGRANT/QFabric/region_all.json"
    output_image_dir = "/home/anxiao/Datasets/MIGRANT/QFabric/region_all"
    os.makedirs(output_image_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    output_json_data = []

    for item in tqdm(data):
        image_path = item["image_path"]
        objects_list = []
        for obj in item["objects"]:
            if is_area_large(obj["hbb"], 10000):
                objects_list.append(obj)
        if len(objects_list) == 0:
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
