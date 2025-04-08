# Dataset Preprocess

## DOTA-V2.0
The DOTA-v2.0 dataset is available at [this website](https://captain-whu.github.io/DOTA/dataset.html).
Download the Training set & the Validation set of DOTA-v1.0 and DOTA-v2.0, and organize the datasets as follows:

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
The DIOR-R dataset is available at [this website](https://gcheng-nwpu.github.io/#Datasets).Download the dataset and organize it as follows (no modification):

```bash
data_root_ori
└── DIOR-R
    ├── Annotations
    ├── ImageSets
    ├── JPEGImages-test
    └── JPEGImages-trainval
```

---

Run the following script to merge the same categories of DOTA-v2.0 and DIOR-R.
```bash
python script/merge_categories.py
```

Run the following scripts to extract images with only one object for the COG task.
```bash
python scripts/extract_img_with_n_obj.py --dataset_name DOTA-v2_0 --n_obj [1]

python scripts/extract_img_with_n_obj.py --dataset_name DIOR-R --n_obj [1]
```

---

Run the following scripts to extract images with 2 ~ 10 objects (or 2 ~ 4 vehicles) for the DG task.
```bash
python scripts/extract_img_with_n_obj.py --dataset_name DOTA-v2_0 --n_obj [2,10]

python scripts/extract_img_with_n_obj.py --dataset_name DIOR-R --n_obj [2,10]

python scripts/extract_img_with_n_obj.py --dataset_name DIOR-R --n_obj [2,4] --include vehicle
```

Run the following script to remove objects.
```bash
python powerpaint/inference_obj_remove.py <args>
```

# Data Filtering

## Diffrence Grounding

Run the following script to visualize the images before and after applying PowerPaint.
```bash
python PowerPaint/visualize.py <args>
```

Inspect the generated "vis_{}" folder and delete the images where objects are wrongly removed by PowerPaint. After human supervision, run the following script to extract final images.
```bash
python after_supervision/difference_grounding.py --dataset_name DOTA-v2_0 --suffix remove_2_to_10

python after_supervision/difference_grounding.py --dataset_name DIOR-R --suffix remove_2_to_10

python after_supervision/difference_grounding.py --dataset_name DIOR-R --suffix remove_2_to_4_vehicle
```

Vehicles in DOTA-v2.0 are not suitable for Difference Grounding due to its relatively small scale.


# Visual Prompt (VP)

Using the above data (only one object in image) for COG task to genrate VP. Run the sciprt
```bash
python visual_prompt/generate_vp.py
```


# In-context Clip

Run the following scripts to extract images with 3 ~ inf objects for the ICG task.
```bash
python scripts/extract_img_with_n_obj.py --dataset_name DOTA-v2_0 --n_obj [3,1000]

python scripts/extract_img_with_n_obj.py --dataset_name DIOR-R --n_obj [3,1000]
```

Run the script to clip and regional images and json file.
```bash
python in_context_clip/generate_region.py
```
