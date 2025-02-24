from shapely.geometry import Polygon


def calculate_iou(horizontal_box, rotated_box):
    x1, y1, x2, y2 = horizontal_box
    horizontal_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    rotated_polygon = Polygon(
        [
            (rotated_box[0], rotated_box[1]),
            (rotated_box[2], rotated_box[3]),
            (rotated_box[4], rotated_box[5]),
            (rotated_box[6], rotated_box[7]),
        ]
    )

    intersection_area = horizontal_polygon.intersection(rotated_polygon).area
    union_area = horizontal_polygon.union(rotated_polygon).area
    if union_area == 0:
        return 0
    iou = intersection_area / union_area

    return iou
