from PIL import Image
from visual_prompt_generator import image_blending


img_path = "00378.jpg"
image = Image.open(img_path)
img_drawn = image_blending(
    image=image,
    shape="mask",
    # bbox_coord=[482, 531, 713, 629],
    bbox_coord=[557, 592, 572, 589, 579, 625, 564, 628],
)

img_drawn.save("output.jpg")
