import os
import math
import json
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from PIL import Image, ImageDraw
from sklearn.cluster import DBSCAN


def cluster_boxes(items, eps=400):
    centers = np.array(
        [
            [
                (item["hbb"][0] + item["hbb"][2]) / 2,
                (item["hbb"][1] + item["hbb"][3]) / 2,
            ]
            for item in items
        ]
    )
    clustering = DBSCAN(eps=eps, min_samples=1).fit(centers)
    labels = clustering.labels_
    clustered = {}
    for label, item in zip(labels, items):
        clustered.setdefault(label, []).append(item)

    return clustered


def adjust_crop_region(items, image_width, image_height, tile_size=512):
    if len(items) == 1:
        x1, y1, x2, y2 = items[0]["hbb"]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        offset_range = tile_size // 4
        random_offset_x = np.random.randint(-offset_range, offset_range + 1)
        random_offset_y = np.random.randint(-offset_range, offset_range + 1)

        x_start = int(center_x - tile_size / 2 + random_offset_x)
        y_start = int(center_y - tile_size / 2 + random_offset_y)

        x_start = max(0, min(x_start, image_width - tile_size))
        y_start = max(0, min(y_start, image_height - tile_size))

        return x_start, y_start, x_start + tile_size, y_start + tile_size
    else:
        x1s, y1s, x2s, y2s = zip(*[item["hbb"] for item in items])
        min_x, min_y = min(x1s), min(y1s)
        max_x, max_y = max(x2s), max(y2s)

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        x_start = int(center_x - tile_size / 2)
        y_start = int(center_y - tile_size / 2)

        x_start = max(0, min(x_start, image_width - tile_size))
        y_start = max(0, min(y_start, image_height - tile_size))

        return x_start, y_start, x_start + tile_size, y_start + tile_size


def crop_image_and_update_items(images, items, crop_region):
    x_start, y_start, x_end, y_end = crop_region
    cropped_images = [image.crop((x_start, y_start, x_end, y_end)) for image in images]

    updated_items = []
    for item in items:
        x1, y1, x2, y2 = item["hbb"]
        if x1 >= x_start and x2 <= x_end and y1 >= y_start and y2 <= y_end:
            new_item = item.copy()
            try:
                new_item["hbb"] = [
                    x1 - x_start,
                    y1 - y_start,
                    x2 - x_start,
                    y2 - y_start,
                ]
                new_item["obb"] = [
                    item["obb"][i] - x_start if i % 2 == 0 else item["obb"][i] - y_start
                    for i in range(8)
                ]
            except:
                return [], []
            updated_items.append(new_item)
        else:
            # print("Item is outside the crop region")
            return [], []

    return cropped_images, updated_items


def draw_obb(draw, obb_points, color="blue"):
    center_x = sum(obb_points[i] for i in range(0, 8, 2)) / 4
    center_y = sum(obb_points[i + 1] for i in range(0, 8, 2)) / 4
    points = [(obb_points[i], obb_points[i + 1]) for i in range(0, 8, 2)]
    points.sort(key=lambda p: (math.atan2(p[1] - center_y, p[0] - center_x)))
    draw.polygon(points, outline=color, width=2, fill=None)


def draw_hbb(draw, hbb, color="red"):
    draw.rectangle(hbb, outline=color, width=2)


def process_images(image_paths, items, png_dir, plot_dir):
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    w, h = images[0].size

    clustered = cluster_boxes(items, eps=400)

    json_data = []
    for group_id, group_items in clustered.items():
        crop_region = adjust_crop_region(group_items, w, h)
        cropped_imgs, updated_items = crop_image_and_update_items(
            images, group_items, crop_region
        )
        if not cropped_imgs:
            continue
        for i, cropped_img in enumerate(cropped_imgs):
            png_save_path = os.path.join(
                png_dir,
                os.path.basename(image_paths[i]).split(".")[0] + f"_{group_id}_{i}.png",
            )
            os.makedirs(os.path.dirname(png_save_path), exist_ok=True)
            cropped_img.save(png_save_path)

            draw = ImageDraw.Draw(cropped_img)
            for item in updated_items:
                draw_hbb(draw, item["hbb"], color="red")
                # draw_obb(draw, item["obb"], color="blue")
            save_dir = os.path.join(
                plot_dir,
                f"{os.path.basename(image_paths[i]).split('.')[0]}_{group_id}",
            )
            save_path = os.path.join(save_dir, f"{i}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cropped_img.save(save_path)

        json_data.append(
            {
                "image_paths": [
                    os.path.join(
                        png_dir,
                        os.path.basename(image_paths[i]).split(".")[0]
                        + f"_{group_id}_{i}.png",
                    )
                    for i in range(len(cropped_imgs))
                ],
                "plot_paths": [
                    os.path.join(
                        save_dir,
                        f"{i}.png",
                    )
                    for i in range(len(cropped_imgs))
                ],
                "objects": updated_items,
            }
        )

    return json_data


def process_images_for_regions(image_path, items, png_dir, plot_dir):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    clustered = cluster_boxes(items, eps=600)

    json_data = []
    for group_id, group_items in clustered.items():
        # if len(group_items) <= 1:
        #     continue
        crop_region = adjust_crop_region(group_items, w, h, tile_size=800)
        cropped_imgs, updated_items = crop_image_and_update_items(
            [image], group_items, crop_region
        )
        if not cropped_imgs:
            continue
        cropped_img = cropped_imgs[0]
        png_save_path = os.path.join(
            png_dir,
            os.path.basename(image_path).split(".")[0] + f"_{group_id}.png",
        )
        cropped_img.save(png_save_path)

        draw = ImageDraw.Draw(cropped_img)
        for item in updated_items:
            draw_hbb(draw, item["hbb"], color="red")
            # draw_obb(draw, item["obb"], color="blue")
        save_path = os.path.join(
            plot_dir,
            os.path.basename(image_path).split(".")[0] + f"_{group_id}.png",
        )
        cropped_img.save(save_path)

        json_data.append(
            {
                "image_path": png_save_path,
                "plot_paths": save_path,
                "objects": updated_items,
            }
        )

    return json_data


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
        image_1 = Image.open(item["plot_paths"][0])
        image_2 = Image.open(item["plot_paths"][-1])
        all_different = True
        for i in range(len(item["objects"])):
            change_status_1 = item["objects"][i]["change_status"][0]
            change_status_2 = item["objects"][i]["change_status"][-1]
            if change_status_1 == change_status_2:
                all_different = False
                break
        if all_different:
            total_width = image_1.width + image_2.width
            max_height = max(image_1.height, image_2.height)
            new_img = Image.new("RGB", (total_width, max_height))
            new_img.paste(image_1, (0, 0))
            new_img.paste(image_2, (image_1.width, 0))

            image_path = os.path.basename(item["image_paths"][0]).replace("_0", "")
            new_img.save(os.path.join(img_out_path, image_path))


if __name__ == "__main__":

    Image.MAX_IMAGE_PIXELS = None
    QFabric_root = "/home/anxiao/Datasets/MIGRANT/QFabric"
    split_json_dir = "/home/anxiao/Datasets/MIGRANT/QFabric/vectors/random-split-1_2023_02_05-11_47_30/COCO"
    dg_png_save_dir = "/home/anxiao/Datasets/MIGRANT/QFabric/dg/png"
    dg_png_plot_dir = "/home/anxiao/Datasets/MIGRANT/QFabric/dg/plot"
    dg_save_json_path = "/home/anxiao/Datasets/MIGRANT/QFabric/dg/label.json"
    icg_png_save_dir = "/home/anxiao/Datasets/MIGRANT/QFabric/icg/png"
    icg_png_plot_dir = "/home/anxiao/Datasets/MIGRANT/QFabric/icg/plot"
    icg_save_json_path = "/home/anxiao/Datasets/MIGRANT/QFabric/icg/label.json"
    os.makedirs(dg_png_save_dir, exist_ok=True)
    os.makedirs(dg_png_plot_dir, exist_ok=True)
    os.makedirs(icg_png_save_dir, exist_ok=True)
    os.makedirs(icg_png_plot_dir, exist_ok=True)

    metadata_json = os.path.join(split_json_dir, "metadata.json")
    with open(metadata_json, "r") as f:
        metadata = json.load(f)

    train_data = natsorted(metadata["dataset"]["train"])
    val_data = metadata["dataset"]["val"]
    change_types = metadata["label:metadata"][0]["options"]
    change_statuses = metadata["label:metadata"][1]["options"]
    urban_types = metadata["label:metadata"][2]["options"]
    geography_types = metadata["label:metadata"][3]["options"]

    json_data = []
    for json_file in tqdm(train_data[: len(train_data) // 2]):
        json_path = os.path.join(split_json_dir, json_file)
        with open(json_path, "r") as f:
            data = json.load(f)

        annotations = []
        for annotation in data["annotations"]:
            try:
                hbb = [
                    int(annotation["bbox"][0]),
                    int(annotation["bbox"][1]),
                    int(annotation["bbox"][0] + annotation["bbox"][2]),
                    int(annotation["bbox"][1] + annotation["bbox"][3]),
                ]
                obb = [int(coord) for coord in annotation["segmentation"][0][:-2]]
            except:
                continue
            change_type = change_types[annotation["properties"][0]["labels"][0]]
            change_status = [
                change_statuses[annotation["properties"][1]["labels"][str(i)][0]]
                for i in range(5)
            ]
            urban_type = urban_types[annotation["properties"][2]["labels"][0]]
            geography_type = geography_types[annotation["properties"][3]["labels"][0]]
            annotations.append(
                {
                    "hbb": hbb,
                    "obb": obb,
                    "change_type": change_type,
                    "change_status": change_status,
                    "urban_type": urban_type,
                    "geography_type": geography_type,
                }
            )

        images = [
            os.path.join(QFabric_root, item["file_name"]) for item in data["images"]
        ]
        json_data.extend(
            process_images(images, annotations, dg_png_save_dir, dg_png_plot_dir)
        )

    with open(dg_save_json_path, "w") as f:
        json.dump(json_data, f, indent=4)

    json_data = []
    for json_file in tqdm(train_data):
        json_path = os.path.join(split_json_dir, json_file)
        with open(json_path, "r") as f:
            data = json.load(f)

        annotations = []
        for annotation in data["annotations"]:
            try:
                hbb = [
                    int(annotation["bbox"][0]),
                    int(annotation["bbox"][1]),
                    int(annotation["bbox"][0] + annotation["bbox"][2]),
                    int(annotation["bbox"][1] + annotation["bbox"][3]),
                ]
                obb = [int(coord) for coord in annotation["segmentation"][0][:-2]]
            except:
                continue
            change_status = change_statuses[
                annotation["properties"][1]["labels"]["4"][0]
            ]
            if change_status in [
                "Prior Construction",
                "Construction Midway",
                "Construction Done",
                "Operational",
            ]:
                annotations.append(
                    {
                        "object_name": "building",
                        "hbb": hbb,
                        "obb": obb,
                    }
                )

        if not annotations:
            continue
        image = os.path.join(QFabric_root, data["images"][-1]["file_name"])
        json_data.extend(
            process_images_for_regions(
                image, annotations, icg_png_save_dir, icg_png_plot_dir
            )
        )

    with open(icg_save_json_path, "w") as f:
        json.dump(json_data, f, indent=4)
    
    img_out_path = dg_png_plot_dir.replace("plot", "vis")
    os.makedirs(img_out_path, exist_ok=True)

    vis_images(dg_save_json_path, img_out_path)
