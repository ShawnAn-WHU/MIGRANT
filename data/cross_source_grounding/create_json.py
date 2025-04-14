import os
import math
import json
from PIL import Image
from tqdm import tqdm


def hbb_or_obb(coords):
    if not coords:
        return []

    def angle(p1, p2):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        return math.degrees(math.atan2(dy, dx))

    angles = [angle(coords[i], coords[(i + 1) % 4]) % 180 for i in range(4)]

    is_hbb = all(abs(a - 0) == 0 or abs(a - 90) == 0 for a in angles)

    if is_hbb:
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        return [x1, y1, x2, y2]
    else:
        return [coord for point in coords for coord in point]


def get_new_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    if "map" in json_path:
        image_dir = json_path.replace("coords_map.json", "png_map")
    else:
        image_dir = json_path.replace("coords.json", "png")
    os.makedirs(image_dir, exist_ok=True)

    new_data = []
    for item in data:
        coords = json.loads(item["coords"])
        new_coords = hbb_or_obb(coords)
        if len(new_coords) == 0:
            continue
        if len(new_coords) == 4:
            bbox = "hbb"
        else:
            bbox = "obb"
        new_item = {
            "image_name": item["id"] + ".png",
            "image_path": image_dir + "/" + item["id"] + ".png",
            bbox: str(new_coords),
        }
        new_data.append(new_item)

    with open(json_path.replace(".json", "_new.json"), "w") as f:
        json.dump(new_data, f, indent=4)


if __name__ == "__main__":

    image_root = "/home/anxiao/Datasets/MIGRANT/CSG"

    # original images
    # for city in os.listdir(image_root):
    #     city_dir = os.path.join(image_root, city)
    #     if not os.path.isdir(city_dir):
    #         continue
    #     if "png" in city:
    #         continue
    #     if "map" in city:
    #         png_save_dir = os.path.join(image_root, "png_map")
    #     else:
    #         png_save_dir = os.path.join(image_root, "png")
    #     os.makedirs(png_save_dir, exist_ok=True)

    #     for image_name in tqdm(os.listdir(city_dir)):
    #         image_path = os.path.join(city_dir, image_name)
    #         if not image_name.endswith(".tif"):
    #             continue
    #         image = Image.open(image_path).convert("RGB")
    #         image.save(os.path.join(png_save_dir, image_name.replace("tif", "png")))

    # plotted images
    # for city in os.listdir(image_root):
    #     city_dir = os.path.join(image_root, city)
    #     if not os.path.isdir(city_dir):
    #         continue
    #     if "_plot" not in city:
    #         continue
    #     sate_plot_dir = os.path.join(city_dir, "tif_plot")
    #     map_plot_dir = os.path.join(city_dir, "map_plot")
    #     sate_png_save_dir = os.path.join(image_root, "png_plot")
    #     map_png_save_dir = os.path.join(image_root, "png_map_plot")
    #     os.makedirs(sate_png_save_dir, exist_ok=True)
    #     os.makedirs(map_png_save_dir, exist_ok=True)

    #     for image_name in tqdm(os.listdir(sate_plot_dir)):
    #         image_path = os.path.join(sate_plot_dir, image_name)
    #         if not image_name.endswith(".tif"):
    #             continue
    #         image = Image.open(image_path).convert("RGB")
    #         image.save(os.path.join(sate_png_save_dir, image_name.replace("tif", "png")))
    #     for image_name in tqdm(os.listdir(map_plot_dir)):
    #         image_path = os.path.join(map_plot_dir, image_name)
    #         if not image_name.endswith(".tif"):
    #             continue
    #         image = Image.open(image_path).convert("RGB")
    #         image.save(os.path.join(map_png_save_dir, image_name.replace("tif", "png")))

    json_path = os.path.join(image_root, "coords.json")
    get_new_json(json_path)
    json_path = os.path.join(image_root, "coords_map.json")
    get_new_json(json_path)
