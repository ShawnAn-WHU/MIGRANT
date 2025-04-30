# This is the script to preprocess the RSOD dataset.
# The RSOD dataset is available at https://github.com/RSIA-LIESMARS-WHU/RSOD-Dataset-
import os
import re
import json
import shutil
from tqdm import tqdm
from natsort import natsorted

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import constants


PATTERN = r"(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)$"


def copy_images(image_dir, output_dir):
    for category_dir in os.listdir(image_dir):
        src_folder = os.path.join(image_dir, category_dir, "JPEGImages")
        for img in tqdm(os.listdir(src_folder)):
            shutil.copy(os.path.join(src_folder, img), output_dir)


def parse_txt(txt_path):
    with open(txt_path, "r") as f:
        lines = f.readlines()

    objects = []
    for line in lines:
        if line == "\n":
            continue
        line = line.strip()
        match = re.search(PATTERN, line)
        if match:
            top_left_0, top_left_1, bottom_right_0, bottom_right_1 = match.groups()
        else:
            raise ValueError(f"Invalid line format: {line}")
        top_left = [
            max(0, int(top_left_0)),
            max(0, int(top_left_1)),
        ]
        bottom_right = [
            max(0, int(bottom_right_0)),
            max(0, int(bottom_right_1)),
        ]
        category = constants.RSOD_CLASSES[txt_path.split("/")[-1].split("_")[0]]
        objects.append(
            {
                "object_name": category,
                "hbb": str(
                    [
                        top_left[0],
                        top_left[1],
                        bottom_right[0],
                        bottom_right[1],
                    ]
                ),
            }
        )
    return objects


def txt_2_json(image_dir, output_dir, json_output_path):
    json_data = []
    for image_name in tqdm(natsorted(os.listdir(output_dir))):
        category = image_name.split("_")[0]
        image_path = os.path.join(output_dir, image_name)
        txt_path = os.path.join(
            image_dir, category, "Annotation", "labels", image_name.replace(".jpg", ".txt")
        )
        if not os.path.exists(txt_path):
            continue
        objects = parse_txt(txt_path)
        json_data.append(
            {
                "image_name": image_name,
                "image_path": image_path,
                "objects": objects,
            }
        )

    with open(json_output_path, "w") as f:
        json.dump(json_data, f, indent=4)


if __name__ == "__main__":

    image_dir = "/home/anxiao/Datasets/RSOD"
    output_dir = "/home/anxiao/Datasets/MIGRANT/RSOD/images"
    json_output_path = "/home/anxiao/Datasets/MIGRANT/RSOD/label.json"
    os.makedirs(output_dir, exist_ok=True)
    with open(json_output_path, "w") as f:
        json.dump([], f, indent=4)

    # copy_images(image_dir, output_dir)
    txt_2_json(image_dir, output_dir, json_output_path)
