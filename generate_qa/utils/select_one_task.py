import os
import json
import random


task = "cvg"
json_dir = "/home/anxiao/Datasets/MIGRANT/sft"
save_json_path = os.path.join(os.path.dirname(json_dir), f"mig_{task}_2k.json")
save_val_json_path = os.path.join(os.path.dirname(json_dir), f"mig_{task}_200_val.json")

mig_2k_data = []
log_entries = []
mig_val_200_data = []
val_log_entries = []

for file_name in os.listdir(json_dir):
    file_path = os.path.join(json_dir, file_name)
    if file_name != f"{task}.json":
        continue
    with open(file_path, "r") as f:
        data = json.load(f)
    sampled_data = random.sample(data, min(2000, len(data)))
    remaining_data = [item for item in data if item not in sampled_data]
    validation_data = random.sample(remaining_data, min(200, len(remaining_data)))
    mig_2k_data.extend(sampled_data)
    mig_val_200_data.extend(validation_data)
    log_entries.append(f"{file_name}: {len(sampled_data)} samples")
    val_log_entries.append(f"{file_name}: {len(validation_data)} samples")

random.shuffle(mig_2k_data)
random.shuffle(mig_val_200_data)
print(f"Total samples collected: {len(mig_2k_data)}")
print(f"Validation samples collected: {len(mig_val_200_data)}")

with open(save_json_path, "w") as f:
    json.dump(mig_2k_data, f, indent=4)
with open(save_val_json_path, "w") as f:
    json.dump(mig_val_200_data, f, indent=4)
