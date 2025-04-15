import os
import json


def replace_trainstation(data):
    if isinstance(data, dict):
        return {key: replace_trainstation(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [replace_trainstation(item) for item in data]
    elif isinstance(data, str):
        return data.replace("trainstation", "train station")
    else:
        return data


input_folder = "/home/anxiao/Datasets/MIGRANT/sft/"
for file_name in os.listdir(input_folder):
    if file_name.endswith(".json"):
        input_file = os.path.join(input_folder, file_name)
        with open(input_file, "r") as f:
            json_data = json.load(f)
    updated_data = replace_trainstation(json_data)

    with open(input_file, "w") as f:
        json.dump(updated_data, f, indent=4)
