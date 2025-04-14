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
    "Expressway-service-area": "expressway service area",
    "baseballfield": "baseball diamond",
    "Expressway-toll-station": "expressway toll station",
    "golffield": "golf field",
    "groundtrackfield": "ground track field",
    "tenniscourt": "tennis court",
}

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
    "trainstation",
    "vehicle",
    "stadium",
    "basketball court",
    "harbor",
    "baseball diamond",
    "swimming pool",
    "windmill"
]

rewrite_prompt = """You are provided with a text. Your task is to rewrite the text using different words while maintaining the same meaning. Produce 50 distinct versions of the text.

# Steps

1. Read and understand the original text.
2. Identify key phrases and their meanings.
3. Use synonyms and alternative phrases to construct a new version.
4. Ensure that the meaning of the original text is preserved in each version.
5. Repeat this process until you create 50 unique texts.

# Output Format

The output should consist of 50 separate text variations. Each version should be distinct but retain the meaning of the original text, ideally formatted in a simple list.

# Notes

Consider that some variations may inherently have different nuances; strive for variety while staying true to the original meaning."""


QWEN2_VL_QA_FORMAT = [
    {
        "role": "user",
        "content": "",
    },
    {
        "role": "assistant",
        "content": "",
    }
]

QWEN2_VL_FORMAT = {
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
    ],
    "images": []
}

