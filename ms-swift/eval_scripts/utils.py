import re
import json
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import box as shaply_box


PATTERN = r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|><\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)(?:,\((\d+),(\d+)\),\((\d+),(\d+)\))?<\|box_end\|>"
CSG_PATTERN = r"<\|box_start\|>\((\d+\.?\d*),\s*(\d+\.?\d*)\),\((\d+\.?\d*),\s*(\d+\.?\d*)\)(?:,\((\d+\.?\d*),\s*(\d+\.?\d*)\),\((\d+\.?\d*),\s*(\d+\.?\d*)\))?<\|box_end\|>"
CVG_PATTERN = r"<\|point\|>\((\d+\.?\d*),\s*(\d+\.?\d*)\)<\|point\|>"
CVG_IMG_PATTERN = r"Image-\d+"
IOU_THREASHOLD = 0.5


def parse_cls_bbox(content):
    detections = []
    for match in re.findall(PATTERN, content):
        cls = match[0]
        coords = list(match[1:])
        detections.append({'class': cls, 'bbox': coords})
    return detections


def split_by_image(content):
    parts = re.split(r'Image-\d+:\s*', content.strip())
    return parts[1:]


def compute_iou(box1, box2):
    box1 = box1[:4] if box1[4] == "" else box1
    box2 = box2[:4] if box2[4] == "" else box2
    def to_polygon(box):
        if len(box) == 4:
            x1, y1, x2, y2 = box
            return shaply_box(int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2)))
        elif len(box) == 8:
            return Polygon([(int(float(box[i])), int(float(box[i + 1]))) for i in range(0, len(box), 2)])
        else:
            raise ValueError("Invalid box format")

    poly1 = to_polygon(box1)
    poly2 = to_polygon(box2)
    if not poly1.is_valid or not poly2.is_valid:
        return 0

    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    iou = intersection / union if union > 0 else 0
    return iou


def compute_metrics(TP, FP, FN, ious):
    accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    miou = sum(ious) / len(ious) if ious else 0
    return accuracy, precision, recall, f1_score, miou


def compute_ed(p1, p2):
    p1 = [float(coord) / 1000 * 640 for coord in p1]
    p2 = [float(coord) / 1000 * 640 for coord in p2]
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)

    return np.linalg.norm(p1 - p2)


def ed_at_thresholds(distances, thresholds=[20, 50, 100, 150]):
    for t in thresholds:
        ratio = (np.asarray(distances) < t).mean()
        print(f"ed < {t}px : {ratio * 100:.2f}%")


def eval_cog_item(response, answer):
    assert response["images"] == answer["images"], "Image list mismatch"
    assert (response["messages"][1::2] == answer["messages"][1::2]), "Message list mismatch"
    num_images = len(response["images"])

    TP, FP, FN, ious = 0, 0, 0, []
    for response_message, answer_message in zip(response["messages"][2::2], answer["messages"][2::2]):
        response_matches = re.findall(PATTERN, response_message["content"])
        answer_matches = re.findall(PATTERN, answer_message["content"])
        if not response_matches:
            FP += num_images if response_message["content"] != answer_message["content"] else 0
            FN += num_images if response_message["content"] != answer_message["content"] else 0
        else:
            for response_match, answer_match in zip(response_matches, answer_matches):
                response_box = list(response_match[1:])
                answer_box = list(answer_match[1:])
                iou = compute_iou(response_box, answer_box)
                ious.append(iou)
                if iou >= IOU_THREASHOLD and response_match[0] == answer_match[0]:
                    TP += 1
                else:
                    FP += 1 if FP < num_images else 0
                    FN += 1 if FN < num_images else 0
    return TP, FP, FN, ious


def eval_cog(response_json, answer_json):
    with open(response_json, "r") as f:
        response_data = json.load(f)
    with open(answer_json, "r") as f:
        answer_data = json.load(f)

    assert len(response_data) == len(answer_data), "Response and answer data length mismatch"

    TP, FP, FN, ious = 0, 0, 0, []
    for response_item, answer_item in zip(response_data, answer_data):
        tp, fp, fn, iou = eval_cog_item(response_item, answer_item)
        TP += tp
        FP += fp
        FN += fn
        ious.extend(iou)
    
    return compute_metrics(TP, FP, FN, ious)


def eval_csg_item(response, answer):
    assert response["images"] == answer["images"], "Image list mismatch"
    assert (response["messages"][1] == answer["messages"][1]), "Message mismatch"

    response_message, answer_message = response["messages"][2], answer["messages"][2]
    response_match = re.findall(CSG_PATTERN, response_message["content"])[0]
    answer_match = re.findall(CSG_PATTERN, answer_message["content"])[0]
    response_box = list(response_match)
    answer_box = list(answer_match)
    iou = compute_iou(response_box, answer_box)
    return iou


def eval_csg(response_json, answer_json):
    with open(response_json, "r") as f:
        response_data = json.load(f)
    with open(answer_json, "r") as f:
        answer_data = json.load(f)

    assert len(response_data) == len(answer_data), "Response and answer data length mismatch"

    ious = []
    for response_item, answer_item in zip(response_data, answer_data):
        iou = eval_csg_item(response_item, answer_item)
        ious.append(iou)

    return sum(ious) / len(ious)


def eval_cvg_item(response, answer):
    assert response["images"] == answer["images"], "Image list mismatch"
    assert (response["messages"][1::2] == answer["messages"][1::2]), "Message list mismatch"

    correct, total, eds = 0, 0, []
    continuous = False
    for response_message, answer_message in zip(response["messages"][2::2], answer["messages"][2::2]):
        response_matches = re.findall(CVG_PATTERN, response_message["content"])
        answer_matches = re.findall(CVG_PATTERN, answer_message["content"])
        if not response_matches:
            img_id_response = re.findall(CVG_IMG_PATTERN, response_message["content"])[0]
            img_id_answer = re.findall(CVG_IMG_PATTERN, answer_message["content"])[0]
            correct += 1 if img_id_response == img_id_answer else 0
            total += 1
            continuous = True
            hit = img_id_response == img_id_answer
        else:
            if continuous and not hit:
                continuous = False
                continue
            elif continuous and hit:
                point_response = list(response_matches[0])
                point_answer = list(answer_matches[0])
                ed = compute_ed(point_response, point_answer)
                eds.append(ed)
                continuous = False
            else:
                img_id_response = re.findall(CVG_IMG_PATTERN, response_message["content"])
                if img_id_response:
                    img_id_response = img_id_response[0]
                    img_id_answer = re.findall(CVG_IMG_PATTERN, answer_message["content"])[0]
                    correct += 1 if img_id_response == img_id_answer else 0
                    total += 1
                    if img_id_response == img_id_answer:
                        point_response = list(response_matches[0])
                        point_answer = list(answer_matches[0])
                        ed = compute_ed(point_response, point_answer)
                        eds.append(ed)

    return correct, total, eds


def eval_cvg(response_json, answer_json):
    with open(response_json, "r") as f:
        response_data = json.load(f)
    with open(answer_json, "r") as f:
        answer_data = json.load(f)

    assert len(response_data) == len(answer_data), "Response and answer data length mismatch"

    corrects, totals, eds = 0, 0, []
    for response_item, answer_item in zip(response_data, answer_data):
        correct, total, ed = eval_cvg_item(response_item, answer_item)
        corrects += correct
        totals += total
        eds.extend(ed)
    
    acc = corrects / totals if totals > 0 else 0
    mean_ed = sum(eds) / len(eds) if eds else 0

    return acc, mean_ed, ed_at_thresholds(eds)


def eval_dg_item(response, answer):
    assert response["images"] == answer["images"], "Image list mismatch"
    assert (response["messages"][1::2] == answer["messages"][1::2]), "Message list mismatch"

    TP, FP, FN, ious = 0, 0, 0, []
    for response_message, answer_message in zip(
        response["messages"][2::2], answer["messages"][2::2]
    ):
        response_matches = re.findall(PATTERN, response_message["content"])
        if not response_matches:
            continue
        response_dets = parse_cls_bbox(response_message["content"])
        answer_dets = parse_cls_bbox(answer_message["content"])
        matched_gt = set()

        for pred in response_dets:
            best_iou = 0
            best_idx = None
            for idx, gt in enumerate(answer_dets):
                if idx in matched_gt or pred["class"] != gt["class"]:
                    continue
                current_iou = compute_iou(pred["bbox"], gt["bbox"])
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_idx = idx
            if best_idx is not None:
                ious.append(best_iou)
                if best_iou >= IOU_THREASHOLD:
                    TP += 1
                    matched_gt.add(best_idx)
                else:
                    FP += 1
            else:
                FP += 1
        FN += len(answer_dets) - len(matched_gt)

    return TP, FP, FN, ious


def eval_dg(response_json, answer_json):
    with open(response_json, "r") as f:
        response_data = json.load(f)
    with open(answer_json, "r") as f:
        answer_data = json.load(f)

    assert len(response_data) == len(answer_data), "Response and answer data length mismatch"

    TP, FP, FN, ious = 0, 0, 0, []
    for response_item, answer_item in zip(response_data, answer_data):
        tp, fp, fn, iou = eval_dg_item(response_item, answer_item)
        TP += tp
        FP += fp
        FN += fn
        ious.extend(iou)

    return compute_metrics(TP, FP, FN, ious)


def eval_icg_item(response, answer):
    assert response["images"] == answer["images"], "Image list mismatch"
    assert (response["messages"][1::2] == answer["messages"][1::2]), "Message list mismatch"
    num_images = len(response["images"]) - 1

    TP, FP, FN, ious = 0, 0, 0, []
    continuous = False
    for response_message, answer_message in zip(response["messages"][2::2], answer["messages"][2::2]):
        response_matches = re.findall(PATTERN, response_message["content"])
        answer_matches = re.findall(PATTERN, answer_message["content"])
        if not response_matches:
            FP += 1 if response_message["content"] != answer_message["content"] else 0
            FN += 1 if response_message["content"] != answer_message["content"] else 0
            continuous = True
        else:
            for response_match, answer_match in zip(response_matches, answer_matches):
                response_box = list(response_match[1:])
                answer_box = list(answer_match[1:])
                iou = compute_iou(response_box, answer_box)
                ious.append(iou)
                if iou >= IOU_THREASHOLD and response_match[0] == answer_match[0]:
                    TP += 1
                else:
                    if continuous and response_match[0] != answer_match[0]:
                        continuous = False
                        continue
                    else:
                        FP += 1 if FP < num_images else 0
                        FN += 1 if FN < num_images else 0
    return TP, FP, FN, ious


def eval_icg(response_json, answer_json):
    with open(response_json, "r") as f:
        response_data = json.load(f)
    with open(answer_json, "r") as f:
        answer_data = json.load(f)

    assert len(response_data) == len(answer_data), "Response and answer data length mismatch"

    TP, FP, FN, ious = 0, 0, 0, []
    for response_item, answer_item in zip(response_data, answer_data):
        tp, fp, fn, iou = eval_icg_item(response_item, answer_item)
        TP += tp
        FP += fp
        FN += fn
        ious.extend(iou)
    
    return compute_metrics(TP, FP, FN, ious)


def eval_ig_item(response, answer):
    assert response["images"] == answer["images"], "Image list mismatch"
    assert (response["messages"][1::2] == answer["messages"][1::2]), "Message list mismatch"

    TP, FP, FN, ious = 0, 0, 0, []
    for response_message, answer_message in zip(
        response["messages"][2::2], answer["messages"][2::2]
    ):
        response_matches = re.findall(PATTERN, response_message["content"])
        if not response_matches:
            continue
        response_segs = split_by_image(response_message["content"])
        answer_segs = split_by_image(answer_message["content"])
        for response_seg, answer_seg in zip(response_segs, answer_segs):
            response_dets = parse_cls_bbox(response_seg)
            answer_dets = parse_cls_bbox(answer_seg)
            matched_gt = set()

            for pred in response_dets:
                best_iou = 0
                best_idx = None
                for idx, gt in enumerate(answer_dets):
                    if idx in matched_gt or pred["class"] != gt["class"]:
                        continue
                    current_iou = compute_iou(pred["bbox"], gt["bbox"])
                    if current_iou > best_iou:
                        best_iou = current_iou
                        best_idx = idx
                if best_idx is not None:
                    ious.append(best_iou)
                    if best_iou >= IOU_THREASHOLD:
                        TP += 1
                        matched_gt.add(best_idx)
                    else:
                        FP += 1
                else:
                    FP += 1
            FN += len(answer_dets) - len(matched_gt)

    return TP, FP, FN, ious


def eval_ig(response_json, answer_json):
    with open(response_json, "r") as f:
        response_data = json.load(f)
    with open(answer_json, "r") as f:
        answer_data = json.load(f)

    assert len(response_data) == len(answer_data), "Response and answer data length mismatch"

    TP, FP, FN, ious = 0, 0, 0, []
    for response_item, answer_item in zip(response_data, answer_data):
        tp, fp, fn, iou = eval_ig_item(response_item, answer_item)
        TP += tp
        FP += fp
        FN += fn
        ious.extend(iou)

    return compute_metrics(TP, FP, FN, ious)


def assign_task_type(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    for item in data:
        images = item["images"]
        images_str = "".join(images)
        if "vp_1_obj" in images_str:
            item["task_type"] = "ig"
        elif "region_3_to_10000_obj_all" in images_str:
            item["task_type"] = "icg"
        elif "VIGOR" in images_str:
            item["task_type"] = "cvg"
        elif "remove_2_to_10" in images_str:
            item["task_type"] = "dg"
        else:
            item["task_type"] = "cog"

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
