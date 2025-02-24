import os
import json
import shutil
import argparse
from tqdm import tqdm


def extract_images(json_path, json_save_path, img_out_path, n):
    # os.makedirs(img_out_path, exist_ok=True)
    with open(json_path, "r") as f:
        res = json.load(f)

    save_res = []
    for item in tqdm(res):
        if len(item["objects"]) in n:
            # image_path = item["image_path"]
            # image_name = item["image_name"]
            # if "." not in image_name:
            #     image_name += ".png"
            # shutil.copy(image_path, os.path.join(img_out_path, image_name))
            save_res.append(item)
        
    with open(json_save_path, "w") as f:
        json.dump(save_res, f, indent=4)


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
        type=str,
        default="[1]",
        help="Number range of objects in the image",
    )
    args = parser.parse_args()

    args.n_obj = json.loads(args.n_obj)
    if len(args.n_obj) == 1:
        num_list = args.n_obj
        out_name = f"{num_list[0]}"
    else:
        num_list = list(range(args.n_obj[0], args.n_obj[1] + 1))
        out_name = f"{num_list[0]}_to_{num_list[-1]}"

    img_out_path = f"/home/anxiao/Datasets/MIGRANT/{args.dataset_name}/{out_name}_obj"
    json_path = f"/home/anxiao/Datasets/MIGRANT/{args.dataset_name}/label.json"
    json_save_path = f"/home/anxiao/Datasets/MIGRANT/{args.dataset_name}/label_{out_name}.json"

    extract_images(json_path, json_save_path, img_out_path, num_list)
