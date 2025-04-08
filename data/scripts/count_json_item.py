import json


json_path = "/home/anxiao/Datasets/MIGRANT/DIOR-R/label_1.json"
with open(json_path, "r") as f:
    data = json.load(f)

print(len(data))
