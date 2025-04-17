import os
from openai import OpenAI

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import constants


client = OpenAI(
    base_url="https://api.openai-proxy.org/v1",
    api_key="sk-AUM1k7fDs1wcRH28o43pkXfByLMru9dvcNSXfSlYP0Fiz6vH",
)

cog_init_query = "Identify the identical object found in these images."
cog_init_bbox_query = "Locate this object in the image by providing its bounding box."
dg_describe_query = "Identify the difference between the two images with a short description."
dg_describe_answer_disappear = "A <object> dissappears in the second image, compared with the first image."
dg_describe_answer_appear = "A <object> appears in the second image, compared with the first image."
dg_bbox = "Give me the coordinates of bounding box where they differ."
dg_hallucination = "There is no difference between the two images."
ig_identify = "Identify the object circled by the <color> ellipse."
icg_identify = "What is the object shown in Region-1?"
csg_bbox = "Identify the highlighted object in <image1>, give me its bounding box coordinates in <image2>."
cvg_match = "Locate where the panomatic image is taken from the satellite image with a point."
cvg_sate2pano = "Determine which panomatic image is taken from the satellite image."
cvg_pano2sate = "Figure out which satellite image the panoramic image was taken from."
cvg_point = "Give me the point coordinates in the satellite image that the panoramic image is taken from."


if __name__ == "__main__":

    rewrite_txt_dir = "rewrite_txt"
    os.makedirs(rewrite_txt_dir, exist_ok=True)

    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": constants.rewrite_prompt},
            {
                "role": "user",
                "content": cvg_point,
            },
        ],
    )

    with open(os.path.join(rewrite_txt_dir, "rewrite_cvg_point.txt"), "w") as f:
        f.write(response.choices[0].message.content)
