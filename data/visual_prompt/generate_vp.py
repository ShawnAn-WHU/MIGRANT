import os
import json
import random
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

    json_path = "/home/anxiao/Datasets/MIGRANT/DIOR-R/label_1.json"
    output_json = "/home/anxiao/Datasets/MIGRANT/DIOR-R/vp_1_obj.json"
    output_image_dir = "/home/anxiao/Datasets/MIGRANT/DIOR-R/vp_1_obj"
    os.makedirs(output_image_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)
    
    output_json_data = []

    for item in tqdm(data):
        image = Image.open(item["image_path"])
        obj = item["objects"][0]    # only one object per image
        vp = random.choice(VP_CATEGORY)
        if vp == "rectangle":
            bbox = random.choice(["hbb", "obb"])
        else:
            bbox = "obb"

        bndbox = json.loads(obj[bbox])
        if len(bndbox) == 4:
            polygon = Polygon([(bndbox[0], bndbox[1]), (bndbox[2], bndbox[1]), (bndbox[2], bndbox[3]), (bndbox[0], bndbox[3])])
        else:
            polygon = Polygon([(bndbox[0], bndbox[1]), (bndbox[2], bndbox[3]), (bndbox[4], bndbox[5]), (bndbox[6], bndbox[7])])
        area = polygon.area
        if area < 400:
            continue

        color, img_drawn = image_blending(
            image=image,
            shape=vp,
            bbox_coord=bndbox,
        )
        img_drawn.save(os.path.join(output_image_dir, f"{item['image_path'].split('/')[-1]}"))

        output_json_item = {
            "image_path_ori": item["image_path"],
            "image_path": os.path.join(output_image_dir, f"{item['image_path'].split('/')[-1]}"),
            "objects": [obj],
            "vp": vp,
            "color": color,
        }

        output_json_data.append(output_json_item)
    
    with open(output_json, "w") as f:
        json.dump(output_json_data, f, indent=4)
