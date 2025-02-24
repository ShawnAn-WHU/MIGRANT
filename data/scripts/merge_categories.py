import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import constants


DOTA_json = "/home/anxiao/Datasets/MIGRANT/DOTA-v2_0/label.json"
DIOR_json = "/home/anxiao/Datasets/MIGRANT/DIOR-R/label.json"

with open(DOTA_json, "r") as f:
    DOTA_data = json.load(f)
with open(DIOR_json, "r") as f:
    DIOR_data = json.load(f)

for item in DOTA_data:
    objects = item["objects"]
    for obj in objects:
        obj["object_name"] = (
            constants.DOTA_DIOR_SYNONYMS[obj["object_name"]]
            if obj["object_name"] in constants.DOTA_DIOR_SYNONYMS
            else obj["object_name"]
        )

with open(DOTA_json, "w") as f:
    json.dump(DOTA_data, f, indent=4)

for item in DIOR_data:
    objects = item["objects"]
    for obj in objects:
        obj["object_name"] = (
            constants.DOTA_DIOR_SYNONYMS[obj["object_name"]]
            if obj["object_name"] in constants.DOTA_DIOR_SYNONYMS
            else obj["object_name"]
        )

with open(DIOR_json, "w") as f:
    json.dump(DIOR_data, f, indent=4)
