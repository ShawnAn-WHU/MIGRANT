import os
import json
import shutil
from tqdm import tqdm


json_path = "/home/anxiao/Datasets/MIGRANT/mig_12k.json"
save_json_path = "/home/anxiao/Datasets/MIGRANT/mig_12k_exp.json"
val_json_path = "/home/anxiao/Datasets/MIGRANT/mig_2k_val.json"
save_val_json_path = "/home/anxiao/Datasets/MIGRANT/mig_2k_val_exp.json"
image_dir = "/home/anxiao/Datasets/MIGRANT"
image_save_dir = "/home/anxiao/Datasets/MIGRANT/12k_exp"
os.makedirs(image_save_dir, exist_ok=True)

with open(val_json_path, "r") as f:
    val_data = json.load(f)

new_val_data = []
for item in tqdm(val_data):
    image_list = item["images"]
    for image_path in image_list:
        save_image_path = image_path.replace(image_dir, image_save_dir)
        os.makedirs(os.path.dirname(save_image_path), exist_ok=True)
        shutil.copy(image_path, save_image_path)
    item["images"] = [image_path.replace(image_dir, image_save_dir) for image_path in image_list]
    new_val_data.append(item)

with open(save_val_json_path, "w") as f:
    json.dump(new_val_data, f, indent=4)   

with open(json_path, "r") as f:
    data = json.load(f)

new_data = []
for item in tqdm(data):
    image_list = item["images"]
    for image_path in image_list:
        save_image_path = image_path.replace(image_dir, image_save_dir)
        os.makedirs(os.path.dirname(save_image_path), exist_ok=True)
        shutil.copy(image_path, save_image_path)
    item["images"] = [image_path.replace(image_dir, image_save_dir) for image_path in image_list]
    new_data.append(item)

with open(save_json_path, "w") as f:
    json.dump(new_data, f, indent=4)     
