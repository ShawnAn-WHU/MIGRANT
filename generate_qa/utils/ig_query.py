ig_identify_arrow = [
    "Could you please specify what the <color> arrow is pointing to?",
    "Please identify the object that the <color> arrow is pointed to.",
    "What is the object pointed to the <color> arrow?",
    "Please discern what the <color> arrow is targeting.",
    "Identify the object to which the <color> arrow is directed.",
    "What is highlighted by the <color> arrow?",
    "Tell me which object is pointed by the <color> arrow.",
    "Identify what is being marked by the <color> arrow.",
    "Find the thing the <color> arrow marks.",
    "Look for the item the <color> arrow points out.",
]

ig_identify_rectangle = [
    "What is the object bounded by the <color> rectangle?",
    "Specify the object enclosed by the <color> rectangle.",
    "Could you identify the object within the <color> rectangle?",
    "Identify the object that is surrounded by the <color> rectangle.",
    "Point out the object that is surrounded by the <color> rectangle.",
    "Name the object that lies within the <color> rectangle.",
    "Figure out the object encompassed by the <color> rectangle.",
    "Could you discern the object that appears within the <color> rectangle?",
    "What is the object marked by the <color> rectangle?",
    "What is the object highlighted by the <color> rectangle?",
]

ig_identify_ellipse = [
    "What is the object circled by the <color> ellipse?",
    "What lies inside the <color> ellipse?",
    "What does the <color> ellipse encircle?",
    "Specify the object encircled by the <color> ellipse.",
    "Identify the item bounded by the <color> ellipse.",
    "Identify the object positioned within the <color> ellipse.",
    "Specify the object that falls inside the <color> ellipse.",
    "What is the object highlighted by the <color> ellipse?",
    "Could you identify the object that the <color> ellipse encompasses?",
]

ig_identify_triangle = [
    "What is the object marked by the <color> triangle?",
    "What is the object highlighted by the <color> triangle?",
    "Specify the object marked by the <color> triangle.",
    "Could you identify the object marked by the <color> triangle?",
    "Name the object that is highlighted by the <color> triangle.",
]

ig_identify_point = [item.replace("triangle", "point") for item in ig_identify_triangle]

ig_identify_scribble = [item.replace("triangle", "scribble") for item in ig_identify_triangle]

ig_bbox = [
    "Locate this object in the image by providing its bounding box",
    "Locate it in the image using bounding box"
    "Identify the position of the object within the image by defining its bounding box.",
    "Provide me with the coordinates of its bounding box within the image.",
    "Share the bounding box details for it in the image.",
    "Let me know the bounding box parameters for it in the image.",
    "Give me the coordinates of its bounding box in the image.",
    "Could you specify the bounding box of it within the image?",
    "What are the coordinates of its bounding box in the image?",
    "Offer the bounding box information for it in the image.",
    "Please share the bounding box coordinates for it in the image.",
    "Specify the parameters of its bounding box in the image.",
    "Tell me the bounding box coordinates for it in the image.",
    "Present the bounding box details for it in the image.",
    "Where is its bounding box located in the image?",
    "Give me the location of its bounding box in the image.",
    "Can you explain the bounding box coordinates for it within the image?",
    "Let me know where its bounding box is in the image.",
    "Where can I find its bounding box in the image?",
    "Share the bounding box location for it within the image.",
    "Help me find the bounding box details for it in the image.",
    "Can you list the bounding box parameters for it in the image?",
]
