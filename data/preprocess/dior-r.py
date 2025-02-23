import os
import json
import shutil
from PIL import Image
from tqdm import tqdm
from natsort import natsorted
import xml.etree.ElementTree as ET


def check_image_size(image_dir):
    def get_image_sizes(image_folder):
        image_paths = [
            os.path.join(image_folder, img) for img in os.listdir(image_folder)
        ]
        return {Image.open(img).size for img in image_paths}

    trainval = os.path.join(image_dir, "JPEGImages-trainval")
    test = os.path.join(image_dir, "JPEGImages-test")

    image_sizes = get_image_sizes(trainval) | get_image_sizes(test)
    print(image_sizes)


def copy_images(image_dir, output_dir):
    for folder in ["JPEGImages-trainval", "JPEGImages-test"]:
        src_folder = os.path.join(image_dir, folder)
        for img in tqdm(os.listdir(src_folder)):
            shutil.copy(os.path.join(src_folder, img), output_dir)


def parse_xml(hbb_xml_path, obb_xml_path):
    tree_hbb = ET.parse(hbb_xml_path)
    root_obb = ET.parse(obb_xml_path)
    root_hbb = tree_hbb.getroot()
    root_obb = root_obb.getroot()

    filename = root_hbb.find("filename").text
    hbb_boxes = []
    obb_boxes = []
    for obj in root_hbb.findall("object"):
        obj_name = obj.find("name").text
        bndboxbox = obj.find("bndbox")
        hbb_box = [
            int(bndboxbox.find("xmin").text),
            int(bndboxbox.find("ymin").text),
            int(bndboxbox.find("xmax").text),
            int(bndboxbox.find("ymax").text),
        ]
        hbb_boxes.append({"name": obj_name, "bbox": str(hbb_box)})
    for obj in root_obb.findall("object"):
        obj_name = obj.find("name").text
        robndbox = obj.find("robndbox")
        obb_box = [
            int(robndbox.find("x_left_top").text),
            int(robndbox.find("y_left_top").text),
            int(robndbox.find("x_right_top").text),
            int(robndbox.find("y_right_top").text),
            int(robndbox.find("x_right_bottom").text),
            int(robndbox.find("y_right_bottom").text),
            int(robndbox.find("x_left_bottom").text),
            int(robndbox.find("y_left_bottom").text),
        ]
        obb_boxes.append({"name": obj_name, "bbox": str(obb_box)})

    return filename, hbb_boxes, obb_boxes


def xml_2_json(image_dir, image_out_path, json_output_path):
    hbb_dir = os.path.join(image_dir, "Annotations", "Horizontal Bounding Boxes")
    obb_dir = os.path.join(image_dir, "Annotations", "Oriented Bounding Boxes")
    hbb_xmls = natsorted(os.listdir(hbb_dir))
    obb_xmls = natsorted(os.listdir(obb_dir))

    json_data = []
    for i in tqdm(range(len(hbb_xmls))):
        hbb_xml_path = os.path.join(hbb_dir, hbb_xmls[i])
        obb_xml_path = os.path.join(obb_dir, obb_xmls[i])
        filename, hbb_boxes, obb_boxes = parse_xml(hbb_xml_path, obb_xml_path)
        item = {
            "image_name": filename,
            "image_path": os.path.join(image_out_path, filename),
        }
        objects = []
        for hbb, obb in zip(hbb_boxes, obb_boxes):
            objects.append(
                {
                    "object_name": obb["name"],
                    "obb": obb["bbox"],
                    "hbb": hbb["bbox"],
                }
            )
        item["objects"] = objects
        json_data.append(item)

    with open(json_output_path, "w") as f:
        json.dump(json_data, f, indent=4)


if __name__ == "__main__":

    dior_r_path = "/home/anxiao/Datasets/DIOR-R"
    # check_image_size(dior_r_path)
    image_output_path = "/home/anxiao/Datasets/MIGRANT/DIOR-R/images"
    json_output_path = "/home/anxiao/Datasets/MIGRANT/DIOR-R/label.json"
    os.makedirs(image_output_path, exist_ok=True)
    with open(json_output_path, "w") as f:
        json.dump([], f, indent=4)

    copy_images(dior_r_path, image_output_path)
    xml_2_json(dior_r_path, image_output_path, json_output_path)
