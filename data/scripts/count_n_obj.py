import os
import json
from tqdm import tqdm


def count_n_obj(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    obj_count_per_image = {}

    for item in tqdm(data):
        num_obj = len(item['objects'])
        if str(num_obj) in obj_count_per_image:
            obj_count_per_image[str(num_obj)] += 1
        else:
            obj_count_per_image[str(num_obj)] = 1
    
    return obj_count_per_image


if __name__ == "__main__":

    DOTA_json = "/home/anxiao/Datasets/MIGRANT/DOTA-v2_0/label.json"
    DIOR_json = "/home/anxiao/Datasets/MIGRANT/DIOR-R/label.json"

    DOTA_obj_count = count_n_obj(DOTA_json)
    DIOR_obj_count = count_n_obj(DIOR_json)
    print(DOTA_obj_count)
    print(DIOR_obj_count)

    # DOTA_obj_count_1 = DOTA_obj_count.get('1', 0)
    # DIOR_obj_count_1 = DIOR_obj_count.get('1', 0)
    # DOTA_obj_count_2_5 = sum([DOTA_obj_count.get(str(i), 0) for i in range(2, 11)])
    # DIOR_obj_count_2_5 = sum([DIOR_obj_count.get(str(i), 0) for i in range(2, 11)])
    # DOTA_obj_count_total = sum(DOTA_obj_count.values())
    # DIOR_obj_count_total = sum(DIOR_obj_count.values())

    # print(f"DOTA-v2_0: 1 object: {DOTA_obj_count_1}, 2-5 objects: {DOTA_obj_count_2_5}, total: {DOTA_obj_count_total}")
    # print(f"DIOR-R: 1 object: {DIOR_obj_count_1}, 2-5 objects: {DIOR_obj_count_2_5}, total: {DIOR_obj_count_total}")