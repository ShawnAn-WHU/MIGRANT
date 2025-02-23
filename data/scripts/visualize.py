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
        draw = ImageDraw.Draw(img)
        draw_objects(draw, item["objects"])
        image_name = item["image_name"]
        if "." not in image_name:
            image_name += ".png"
        img.save(os.path.join(img_out_path, image_name))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        "-n",
        type=str,
        default="DIOR-R",
        help="Name of the dataset",
    )
    parser.add_argument(
        "--n_obj",
        type=int,
        default=0,
        help="Number of objects in the image",
    )
    args = parser.parse_args()

    if not args.n_obj:
        img_out_path = f"/home/anxiao/Datasets/MIGRANT/{args.dataset_name}/vis"
        json_path = f"/home/anxiao/Datasets/MIGRANT/{args.dataset_name}/label.json"
    else:
        img_out_path = (
            f"/home/anxiao/Datasets/MIGRANT/{args.dataset_name}/vis_{args.n_obj}_obj"
        )
        json_path = (
            f"/home/anxiao/Datasets/MIGRANT/{args.dataset_name}/label_{args.n_obj}.json"
        )

    vis_images(json_path, img_out_path)
