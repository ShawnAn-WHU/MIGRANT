import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        "-j",
        type=str,
        default="results_mig_2k_val_v1_1.json",
        help="Path to the JSON file to be processed.",
    )
    args = parser.parse_args()

    with open(args.json_path, "r") as f:
        data = json.load(f)

    for item in data:
        images = item["images"]
        images_str = "".join(images)
        if "vp_1_obj" in images_str:
            item["task_type"] = "ig"
        elif "region_3_to_10000_obj_all" in images_str:
            item["task_type"] = "icg"
        elif "VIGOR" in images_str:
            item["task_type"] = "cvg"
        elif "remove_2_to_10" in images_str:
            item["task_type"] = "dg"
        else:
            item["task_type"] = "cog"

    with open(args.json_path, "w") as f:
        json.dump(data, f, indent=4)
