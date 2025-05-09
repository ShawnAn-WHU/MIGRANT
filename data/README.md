# 1. Dataset Preprocess
> Move images to one folder and unify the format of labels for each dataset.

## DOTA-V2.0
The DOTA-v2.0 dataset is available at [this website](https://captain-whu.github.io/DOTA/dataset.html). Download the Training set & the Validation set of DOTA-v1.0 and DOTA-v2.0, and organize the datasets as follows:

```bash
data_root_ori
└── DOTA-v2.0
    ├── train
    │   ├── images
    │   │   ├── images  # unzip part1/2/3 of DOTA-v1.0 --> automaticaly merge into ./images
    │   │   ├── part4
    │   │   ├── part5
    │   │   └── part6
    │   └── labelTxt-v2.0
    │       ├── DOTA-v2.0_train
    │       └── DOTA-v2.0_train_hbb
    └── val
        ├── images
        │   ├── images  # unzip part1 of DOTA-v1.0 --> automaticaly merge into ./images
        │   └── part2
        └── labelTxt-v2.0
            ├── DOTA-v2.0_val
            └── DOTA-v2.0_val_hbb
```

---

## DIOR-R
The DIOR-R dataset is available at [this website](https://gcheng-nwpu.github.io/#Datasets). Download the dataset and organize it as follows (no modification):

```bash
data_root_ori
└── DIOR-R
    ├── Annotations
    ├── ImageSets
    ├── JPEGImages-test
    └── JPEGImages-trainval
```

---

## NWPU-VHR-10
The NWPU-VHR-10 dataset is available at https://github.com/Gaoshuaikun/NWPU-VHR-10. Download the dataset and organize it as follows (no modification):

```bash
data_root_ori
└── NWPU-VHR-10
    ├── ground truth
    ├── negative image set
    ├── positive image set
    └── readme.txt
```

---

## RSOD
The RSOD dataset is available at https://github.com/RSIA-LIESMARS-WHU/RSOD-Dataset-. Download the dataset and organize it as follows (no modification):

```bash
data_root_ori
└── RSOD
    ├── aircraft
    ├── oiltank
    ├── overpass
    └── playground
```

Run the following scripts to complete this step:

```bash
python dota-v2_0.py
python dior-r.py
python nwpu-vhr-10.py
python rsod.py
```

---

## 🛠️ Preprocess Workflow

Firstly, unify the category sets from these four datasets to establish a comprehensive list of object categories for further processing. For the final category definitions, please refer to [./utils/constants.py](./utils/constants.py). Run the following script to complete this step:

```bash
python scripts/merge_categories.py
```

---

# 2. Data Selection for Tasks
We have obtained the images and label.json of each dataset after preprocess. Then different images (_i.e.,_ images with different objects) need selecting for different tasks.


## Common Object Grounding (COG)
> Ground the same object that appears in every image.<br>
> Requirements: The image contains only one targeted object.

Run the following scripts to extract images with only one object for the COG task.

```bash
python scripts/extract_img_with_n_obj.py --dataset_name DOTA-v2_0 --n_obj [1]

python scripts/extract_img_with_n_obj.py --dataset_name DIOR-R --n_obj [1]
```

## Difference Grounding (DG)
> Ground the vanished objects between two images.<br>
> Requirements: The image contains 2 ~ 10 targeted object or 2 ~ 4 vehicles.<br>
> Note: We exclude the vehicles in DOTA-v2_0 because they are not ideal for this task (relatively small scale). And the vehicles are separatly considered in PowerPaint for better removing performance.

Run the following scripts to extract images with specific numbers of objects for the DG task.

```bash
python scripts/extract_img_with_n_obj.py --dataset_name DOTA-v2_0 --n_obj [2,10]

python scripts/extract_img_with_n_obj.py --dataset_name DIOR-R --n_obj [2,10]

# only extract vehicles
python scripts/extract_img_with_n_obj.py --dataset_name DIOR-R --n_obj [2,4] --include vehicle
```

Run the following script to remove vehicles.

```bash
python PowerPaint/inference_obj_remove_vehicle.py \
--input_json /home/anxiao/Datasets/MIGRANT/DIOR-R/label_2_to_4_include_vehicle.json \
--output_dir /home/anxiao/Datasets/MIGRANT/DIOR-R/remove_2_to_4_vehicle \
--output_json /home/anxiao/Datasets/MIGRANT/DIOR-R/label_remove_2_to_4_vehicle.json
```

Run the following script to remove other objects. In order to avoid image repeatation, we exclude the images that are used in vehicles. So we add an extra argument --refer_json.

```bash
python PowerPaint/inference_obj_remove.py \
--input_json /home/anxiao/Datasets/MIGRANT/DIOR-R/label_2_to_10.json \
--output_dir /home/anxiao/Datasets/MIGRANT/DIOR-R/remove_2_to_10 \
--output_json /home/anxiao/Datasets/MIGRANT/DIOR-R/label_remove_2_to_10.json
--refer_json /home/anxiao/Datasets/MIGRANT/DIOR-R/label_remove_2_to_4_vehicle.json
```

For /home/anxiao/Datasets/MIGRANT/DOTA-v2_0/label_2_to_10.json, you need to **uncomment** the --refer_json part.

### 👨 Human Filtering

PowerPaint cannot accuraly remove the objects all the time. Therefore, it is neccessary to cherry-pick the ideal origin-removed image pairs by hand. Run the following script to visualize the origin-removed image pairs for facillitating the filtering process.

```bash
python PowerPaint/visualize.py --dataset_name DIOR-R --suffix remove_2_to_4_vehicle
```

Inspect the generated "vis_{suffix}" folder and directly delete the images where objects are wrongly removed by PowerPaint. After human supervision, run the following script to extract final images and "label_remove_{suffix}.json" files.

```bash
python after_supervision/difference_grounding.py --dataset_name DIOR-R --suffix remove_2_to_4_vehicle

python after_supervision/difference_grounding.py --dataset_name DIOR-R --suffix remove_2_to_10

python after_supervision/difference_grounding.py --dataset_name DOTA-v2_0 --suffix remove_2_to_10
```

# In-context Grounding (ICG)
> Ground the regional objects by the cropped images.

Run the following scripts to extract images with 3 ~ inf objects for the ICG task.

```bash
python scripts/extract_img_with_n_obj.py --dataset_name DOTA-v2_0 --n_obj [3,10000]

python scripts/extract_img_with_n_obj.py --dataset_name DIOR-R --n_obj [3,10000]
```

Run the script to clip and regional images and json file.
```bash
python in_context_clip/generate_region.py <args>
```

## Iterative Grounding (IG)
> Ground the objects prompted visually (_e.g.,_ arrow, triangle, scribble).<br>
> Requirements: The image contains 2 ~ 4 targeted objects (but generate only one visual prompt).<br>

Run the following scripts to extract images with specific numbers of objects:

```bash
python scripts/extract_img_with_n_obj.py --dataset_name DOTA-v2_0 --n_obj [1,4]

python scripts/extract_img_with_n_obj.py --dataset_name DIOR-R --n_obj [1,4]
```

Run the following script to generate visual prompt:
```bash
python visual_prompt/generate_vp.py --dataset_name DIOR-R --suffix 1_to_4
```

### 👨 Human Filtering
Some visual prompts may be not suitable, so human filtering is also neede. Inspect the generated "vp_{suffix}" folder and directly delete the images where visual prompts are not ideal. After human supervision, run the following script to extract final images and "vp_{suffix}_obj.json" file.

```bash
python after_supervision/difference_grounding.py --dataset_name DIOR-R --suffix vp_1_to_4_obj
```
