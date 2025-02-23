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

run the following script to extract images with only one object for the COG task.
```bash
python scripts/extract_img_with_n_obj.py --dataset_name DOTA-v2_0 --n_obj 1
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

run the following script to extract images with only one object for the COG task.
```bash
python scripts/extract_img_with_n_obj.py --dataset_name DIOR-R --n_obj 1
```

run the following script to merge the same categories of DOTA-v2.0 and DIOR-R.
```bash
python script/merge_categories.py
```