# This is the script to preprocess the DOTA-v2.0 dataset.
# Modified from https://github.com/CAPTAIN-WHU/DOTA_devkit
# The DOTA-v2.0 dataset is available at https://captain-whu.github.io/DOTA/dataset.html
import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import shapely.geometry as shgeo

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import dota_utils


def choose_best_pointorder_fit_another(poly1, poly2):
    """Choose the best point order fit another polygon

    Args:
        poly1 (list): modified polygon
        poly2 (list): reference polygon
    """

    combinate = [
        poly1,
        poly1[2:] + poly1[:2],
        poly1[4:] + poly1[:4],
        poly1[6:] + poly1[:6],
    ]
    dst_coordinate = np.array(poly2)
    distances = [np.sum((np.array(coord) - dst_coordinate) ** 2) for coord in combinate]
    return combinate[np.argmin(distances)]


def cal_line_length(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


class splitbase:
    def __init__(
        self,
        basepath,
        outpath,
        gap=100,
        subsize=1024,
        thresh=0.8,
        choosebestpoint=True,
        initjson=False,
        ext=".png",
    ):
        """
        Args:
            basepath (str): base path for dota dataset
            outpath (str): output base path for splited dota dataset,
            gap (int): overlap between two patches
            subsize (int): subsize of patch
            thresh (float): the thresh determine whether to keep the instance if the instance is cut down
            choosebestpoint (bool): used to choose the first point for the cut instance
            initjson (bool): used to initialize json
            ext (str): ext for the image format
        """

        self.basepath = basepath
        self.outpath = outpath
        self.gap = gap
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.thresh = thresh
        self.imagepath = os.path.join(self.basepath, "images")
        labelTxtpaths = os.listdir(os.path.join(self.basepath, "labelTxt-v2.0"))
        for path in labelTxtpaths:
            if "hbb" in path:
                self.labelpath_hbb = os.path.join(self.basepath, "labelTxt-v2.0", path)
            else:
                self.labelpath_obb = os.path.join(self.basepath, "labelTxt-v2.0", path)
        self.outimagepath = os.path.join(self.outpath, "images")
        self.outjsonpath = os.path.join(self.outpath, "label.json")
        self.choosebestpoint = choosebestpoint
        self.initjson = initjson
        self.ext = ext
        os.makedirs(self.outimagepath, exist_ok=True)
        if self.initjson:
            with open(self.outjsonpath, "w") as f:
                json.dump([], f, indent=4)

    def polyorig2sub(self, left, up, poly):
        """Adjusts the coordinates of a polygon by subtracting the given offsets.

        Args:
            left (int): The left offset to subtract from the x-coordinates.
            up (int): The up offset to subtract from the y-coordinates.
            poly (np.ndarray): A 1D numpy array containing the coordinates of the polygon.
        """

        polyInsub = np.zeros(len(poly))
        for i in range(int(len(poly) / 2)):
            polyInsub[i * 2] = int(poly[i * 2] - left)
            polyInsub[i * 2 + 1] = int(poly[i * 2 + 1] - up)
        return polyInsub

    def calchalf_iou(self, poly1, poly2):
        """Calculate the IoU of two polygons relative to the first polygon.
        Args:
            poly1 (shgeo.Polygon): The first polygon.
            poly2 (shgeo.Polygon): The second polygon.

        Returns:
            tuple: A tuple containing the intersection polygon and the IoU value.
        """

        inter_poly = poly1.intersection(poly2)
        half_iou = inter_poly.area / poly1.area
        return inter_poly, half_iou

    def saveimagepatches(self, img, subimgname, left, up):
        subimg = img[up : up + self.subsize, left : left + self.subsize].copy()
        outdir = os.path.join(self.outimagepath, subimgname + self.ext)
        cv2.imwrite(outdir, subimg)

    def GetPoly4FromPoly5(self, poly):
        """Convert a 5-point polygon to a 4-point polygon by merging the shortest edge.

        Args:
            poly (list): A list of 10 coordinates representing a 5-point polygon.

        Returns:
            list: A list of 8 coordinates representing a 4-point polygon.
        """

        distances = [
            cal_line_length(
                (poly[i * 2], poly[i * 2 + 1]),
                (poly[(i + 1) * 2], poly[(i + 1) * 2 + 1]),
            )
            for i in range(4)
        ]
        distances.append(cal_line_length((poly[0], poly[1]), (poly[8], poly[9])))
        pos = np.argmin(distances)
        outpoly = []
        for i in range(5):
            if i == pos:
                outpoly.append((poly[i * 2] + poly[(i * 2 + 2) % 10]) / 2)
                outpoly.append((poly[i * 2 + 1] + poly[(i * 2 + 3) % 10]) / 2)
            elif i != (pos + 1) % 5:
                outpoly.extend(poly[i * 2 : i * 2 + 2])
        return outpoly

    def savepatches(
        self, resizeimg, objects_obb, objects_hbb, subimgname, left, up, right, down
    ):
        imgpoly = shgeo.Polygon([(left, up), (right, up), (right, down), (left, down)])
        with open(self.outjsonpath, "r") as f_out:
            json_data = json.load(f_out)

        item = {
            "image_name": subimgname,
            "image_path": os.path.join(self.outimagepath, subimgname + self.ext),
            "objects": [],
        }

        for obj_obb, obj_hbb in zip(objects_obb, objects_hbb):
            gtpoly = shgeo.Polygon(
                [
                    (obj_obb["poly"][0], obj_obb["poly"][1]),
                    (obj_obb["poly"][2], obj_obb["poly"][3]),
                    (obj_obb["poly"][4], obj_obb["poly"][5]),
                    (obj_obb["poly"][6], obj_obb["poly"][7]),
                ]
            )
            if gtpoly.area <= 0:
                continue

            inter_poly, half_iou = self.calchalf_iou(gtpoly, imgpoly)

            if half_iou == 1:
                polyInsub = self.polyorig2sub(left, up, obj_obb["poly"])
                polyInsub_hbb = self.polyorig2sub(left, up, obj_hbb["bndbox"])
            elif half_iou > self.thresh:
                inter_poly = shgeo.polygon.orient(inter_poly, sign=1)
                out_poly = list(inter_poly.exterior.coords)[0:-1]

                if len(out_poly) < 4:
                    continue
                out_poly2 = [coord for point in out_poly for coord in point]
                if len(out_poly) == 5:
                    out_poly2 = self.GetPoly4FromPoly5(out_poly2)
                elif len(out_poly) > 5:
                    continue

                if self.choosebestpoint:
                    out_poly2 = choose_best_pointorder_fit_another(
                        out_poly2, obj_obb["poly"]
                    )

                polyInsub = [max(1, min(self.subsize - 1, int(x))) for x in out_poly2]
                polyInsub_hbb = self.polyorig2sub(left, up, obj_hbb["bndbox"])
            else:
                continue

            polyInsub = [max(1, min(self.subsize - 1, int(x))) for x in polyInsub]
            polyInsub_hbb = [
                max(1, min(self.subsize - 1, int(x))) for x in polyInsub_hbb
            ]

            obj_item = {
                "object_name": obj_obb["name"],
                "obb": str(polyInsub),
                "hbb": str(polyInsub_hbb),
            }
            item["objects"].append(obj_item)

        if item["objects"]:
            json_data.append(item)
            with open(self.outjsonpath, "w") as f_out:
                json.dump(json_data, f_out, indent=4)
            self.saveimagepatches(resizeimg, subimgname, left, up)

    def SplitSingle(self, name, rate, extent):
        """Split a single image and its ground truth annotations.

        Args:
            name (str): The name of the image file (without extension).
            rate (float): The resize scale for the image.
            extent (str): The image file extension (e.g., '.png').
        """

        img = cv2.imread(os.path.join(self.imagepath, name + extent))
        if img is None:
            return

        fullname_obb = os.path.join(self.labelpath_obb, name.split("/")[-1] + ".txt")
        fullname_hbb = os.path.join(self.labelpath_hbb, name.split("/")[-1] + ".txt")
        objects_obb = dota_utils.parse_dota_poly2(fullname_obb)
        objects_hbb = dota_utils.parse_dota_rec(fullname_hbb)

        for obj in objects_obb:
            obj["poly"] = [rate * x for x in obj["poly"]]
        for obj in objects_hbb:
            obj["bndbox"] = [rate * x for x in obj["bndbox"]]

        resizeimg = (
            cv2.resize(img, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
            if rate != 1
            else img
        )

        outbasename = f"{name.split('/')[-1]}_{rate}_"
        height, width = resizeimg.shape[:2]

        for left in range(0, width - self.subsize + 1, self.slide):
            for up in range(0, height - self.subsize + 1, self.slide):
                right = min(left + self.subsize, width)
                down = min(up + self.subsize, height)
                subimgname = f"{outbasename}{left}_{up}"
                self.savepatches(
                    resizeimg,
                    objects_obb,
                    objects_hbb,
                    subimgname,
                    left,
                    up,
                    right,
                    down,
                )
                if up + self.subsize >= height:
                    break
            if left + self.subsize >= width:
                break

    def splitdata(self, rate):
        imagelist = dota_utils.GetFileFromThisRootDir(self.imagepath)
        imagenames = [
            dota_utils.custombasename(x)
            for x in imagelist
            if dota_utils.custombasename(x) != "Thumbs"
        ]
        for name in tqdm(imagenames):
            self.SplitSingle(name, rate, self.ext)


if __name__ == "__main__":

    split_val = splitbase(
        basepath="/home/anxiao/Datasets/DOTA-v2_0/val",
        outpath="/home/anxiao/Datasets/MIGRANT/DOTA-v2_0",
        initjson=True,
    )

    split_train = splitbase(
        basepath="/home/anxiao/Datasets/DOTA-v2_0/train",
        outpath="/home/anxiao/Datasets/MIGRANT/DOTA-v2_0",
        initjson=False,
    )

    split_val.splitdata(rate=1)
    split_train.splitdata(rate=1)
