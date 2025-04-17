import os
import json
import random


json_dir = "/home/anxiao/Datasets/MIGRANT/sft"
save_json_path = os.path.join(os.path.dirname(json_dir), "mig_12k.json")
save_val_json_path = os.path.join(os.path.dirname(json_dir), "mig_2k_val.json")
log_txt_path = os.path.join(os.path.dirname(json_dir), "mig_12k.txt")
val_log_txt_path = os.path.join(os.path.dirname(json_dir), "mig_2k_val.txt")

mig_12k_data = []
log_entries = []
mig_val_2k_data = []
val_log_entries = []

for file_name in os.listdir(json_dir):
    file_path = os.path.join(json_dir, file_name)
    with open(file_path, "r") as f:
        data = json.load(f)
    sampled_data = random.sample(data, min(2000, len(data)))
    remaining_data = [item for item in data if item not in sampled_data]
    validation_data = random.sample(remaining_data, min(400, len(remaining_data)))
    mig_12k_data.extend(sampled_data)
    mig_val_2k_data.extend(validation_data)
    log_entries.append(f"{file_name}: {len(sampled_data)} samples")
    val_log_entries.append(f"{file_name}: {len(validation_data)} samples")

random.shuffle(mig_12k_data)
random.shuffle(mig_val_2k_data)
print(f"Total samples collected: {len(mig_12k_data)}")
print(f"Validation samples collected: {len(mig_val_2k_data)}")

with open(save_json_path, "w") as f:
    json.dump(mig_12k_data, f, indent=4)
with open(save_val_json_path, "w") as f:
    json.dump(mig_val_2k_data, f, indent=4)
with open(log_txt_path, "w") as log_file:
    log_file.write("\n".join(log_entries))
with open(val_log_txt_path, "w") as val_log_file:
    val_log_file.write("\n".join(val_log_entries))
