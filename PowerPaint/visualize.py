import os
import json
import argparse
from tqdm import tqdm
from PIL import Image, ImageDraw


def draw_objects(draw, objects):
    for obj in objects:
        if "obb" in obj:
            poly = json.loads(obj["obb"])
            draw.polygon(poly, outline="red")
        if "hbb" in obj:
            poly = json.loads(obj["hbb"])
            draw.rectangle(poly, outline="red")


def vis_images(json_path, img_out_path):
    os.makedirs(img_out_path, exist_ok=True)
    with open(json_path, "r") as f:
        res = json.load(f)

    for item in tqdm(res):
        img = Image.open(item["image_path"])
        image_path = item["image_path"].split("/")[-1]
        if os.path.exists(os.path.join(img_out_path, image_path)):
            continue
        img_remove = Image.open(item["output_image_path"])
        draw = ImageDraw.Draw(img)
        draw_remove = ImageDraw.Draw(img_remove)
        draw_objects(draw, item["objects"])
        draw_objects(draw_remove, item["objects"])

        total_width = img.width + img_remove.width
        max_height = max(img.height, img_remove.height)
        new_img = Image.new("RGB", (total_width, max_height))
        new_img.paste(img, (0, 0))
        new_img.paste(img_remove, (img.width, 0))

        image_path = item["image_path"].split("/")[-1]
        new_img.save(os.path.join(img_out_path, image_path))


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
        default="remove_2_to_4_vehicle",
        help="suffix of the dataset",
    )
    args = parser.parse_args()

    img_out_path = (
        f"/home/anxiao/Datasets/MIGRANT/{args.dataset_name}/vis_{args.suffix}"
    )
    json_path = (
        f"/home/anxiao/Datasets/MIGRANT/{args.dataset_name}/label_{args.suffix}.json"
    )

    vis_images(json_path, img_out_path)
