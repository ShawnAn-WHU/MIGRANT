# This is the script to preprocess the NWPU-VHR-10 dataset.
# The NWPU-VHR-10 dataset is available at https://github.com/Gaoshuaikun/NWPU-VHR-10
import os
import json
import shutil
from tqdm import tqdm
from natsort import natsorted

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import constants


def copy_images(image_dir, output_dir):
    src_folder = os.path.join(image_dir, "positive image set")
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
        top_left_0, top_left_1, bottom_right_0, bottom_right_1, label = line.split(",")
        top_left = [
            int(top_left_0.strip().strip("(")),
            int(top_left_1.strip().strip(")")),
        ]
        bottom_right = [
            int(bottom_right_0.strip().strip("(")),
            int(bottom_right_1.strip().strip(")")),
        ]
        category = constants.NWPU_VHR_10_CLASSES[int(label)]
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
        image_path = os.path.join(output_dir, image_name)
        txt_path = os.path.join(
            image_dir, "ground truth", image_name.replace(".jpg", ".txt")
        )
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

    image_dir = "/home/anxiao/Datasets/NWPU-VHR-10"
    output_dir = "/home/anxiao/Datasets/MIGRANT/NWPU-VHR-10/images"
    json_output_path = "/home/anxiao/Datasets/MIGRANT/NWPU-VHR-10/label.json"
    os.makedirs(output_dir, exist_ok=True)
    with open(json_output_path, "w") as f:
        json.dump([], f, indent=4)

    copy_images(image_dir, output_dir)
    txt_2_json(image_dir, output_dir, json_output_path)
