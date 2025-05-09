import os
import json
import argparse


def remove_image(image_dir, vis_dir):
    vis_image_list = os.listdir(vis_dir)
    print(f"after supervision: {len(vis_image_list)} images")

    for image_name in os.listdir(image_dir):
        if image_name not in vis_image_list:
            os.remove(os.path.join(image_dir, image_name))

    print(f"after remove: {len(os.listdir(image_dir))} images")


def remove_json_item(image_dir, json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    print(f"before remove: {len(data)} items")

    data = [
        item
        for item in data
        if item["image_path"].split("/")[-1] in os.listdir(image_dir)
    ]

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"after remove: {len(data)} items")


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
        default="vp_1_obj",
        help="suffix of the dataset",
    )
    args = parser.parse_args()

    root = "/home/anxiao/Datasets/MIGRANT"
    dataset_dir = os.path.join(root, args.dataset_name)
    image_dir = os.path.join(dataset_dir, args.suffix)

    json_path = os.path.join(dataset_dir, f"{args.suffix}.json")
    remove_json_item(image_dir, json_path)
