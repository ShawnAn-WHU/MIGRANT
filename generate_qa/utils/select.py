import os
import json
import random


json_dir = "/home/anxiao/Datasets/MIGRANT/sft"
save_json_path = os.path.join(os.path.dirname(json_dir), "mig_5k.json")

mig_5k_data = []
for file_name in os.listdir(json_dir):
    file_path = os.path.join(json_dir, file_name)
    with open(file_path, "r") as f:
        data = json.load(f)
    mig_5k_data.extend(random.sample(data, 1000))

random.shuffle(mig_5k_data)
with open(save_json_path, "w") as f:
    json.dump(mig_5k_data, f, indent=4)
