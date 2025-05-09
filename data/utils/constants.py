# determine the synonyms of these two datasets
DOTA_DIOR_SYNONYMS = {
    "plane": "airplane",
    "baseball-diamond": "baseball diamond",
    "basketball-court": "basketball court",
    "container-crane": "container crane",
    "ground-track-field": "ground track field",
    "large-vehicle": "vehicle",
    "small-vehicle": "vehicle",
    "soccer-ball-field": "soccer ball field",
    "storage-tank": "storage tank",
    "swimming-pool": "swimming pool",
    "tennis-court": "tennis court",
    "storagetank": "storage tank",
    "basketballcourt": "basketball court",
    "Expressway-Service-area": "expressway service area",
    "baseballfield": "baseball diamond",
    "Expressway-toll-station": "expressway toll station",
    "golffield": "golf field",
    "groundtrackfield": "ground track field",
    "tenniscourt": "tennis court",
}

# final merged categories for DOTA and DIOR
DOTA_DIOR_COMBINE = [
    "airport",
    "container crane",
    "tennis court",
    "airplane",
    "ship",
    "dam",
    "helicopter",
    "storage tank",
    "roundabout",
    "bridge",
    "chimney",
    "ground track field",
    "expressway toll station",
    "golf field",
    "overpass",
    "soccer ball field",
    "helipad",
    "expressway service area",
    "train station",
    "vehicle",
    "stadium",
    "basketball court",
    "harbor",
    "baseball diamond",
    "swimming pool",
    "windmill"
]

# included in DOTA_DIOR_COMBINE
NWPU_VHR_10_CLASSES = {
    1: "airplane",
    2: "ship",
    3: "storage tank",
    4: "baseball diamond",
    5: "tennis court",
    6: "basketball court",
    7: "ground track field",
    8: "harbor",
    9: "bridge",
    10: "vehicle",
}

# included in DOTA_DIOR_COMBINE
RSOD_CLASSES = {
    "aircraft": "airplane",
    "oiltank": "storage tank",
    "playground": "ground track field",
    "overpass": "overpass",
}
