import numpy as np
from utils import parse


def nms(
    bounding_boxes,
    confidence_score,
    labels,
    threshold,
    input_in_pixels=False,
    return_array=True,
):
    """
    This NMS processes boxes of all labels. It not only removes the box with the same label.

    Adapted from https://github.com/amusi/Non-Maximum-Suppression/blob/master/nms.py
    """
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_labels = []

    # Compute areas of bounding boxes
    if input_in_pixels:
        areas = (end_x - start_x + 1) * (end_y - start_y + 1)
    else:
        areas = (end_x - start_x) * (end_y - start_y)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        picked_labels.append(labels[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        if input_in_pixels:
            w = np.maximum(0.0, x2 - x1 + 1)
            h = np.maximum(0.0, y2 - y1 + 1)
        else:
            w = np.maximum(0.0, x2 - x1)
            h = np.maximum(0.0, y2 - y1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    if return_array:
        picked_boxes, picked_score, picked_labels = (
            np.array(picked_boxes),
            np.array(picked_score),
            np.array(picked_labels),
        )

    return picked_boxes, picked_score, picked_labels


def class_aware_nms(
    bounding_boxes, confidence_score, labels, threshold, input_in_pixels=False
):
    """
    This NMS processes boxes of each label individually.
    """
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    picked_boxes, picked_score, picked_labels = [], [], []

    labels_unique = np.unique(labels)
    for label in labels_unique:
        bounding_boxes_label = [
            bounding_box
            for i, bounding_box in enumerate(bounding_boxes)
            if labels[i] == label
        ]
        confidence_score_label = [
            confidence_score_item
            for i, confidence_score_item in enumerate(confidence_score)
            if labels[i] == label
        ]
        labels_label = [label] * len(bounding_boxes_label)
        picked_boxes_label, picked_score_label, picked_labels_label = nms(
            bounding_boxes_label,
            confidence_score_label,
            labels_label,
            threshold=threshold,
            input_in_pixels=input_in_pixels,
            return_array=False,
        )
        picked_boxes += picked_boxes_label
        picked_score += picked_score_label
        picked_labels += picked_labels_label

    picked_boxes, picked_score, picked_labels = (
        np.array(picked_boxes),
        np.array(picked_score),
        np.array(picked_labels),
    )

    return picked_boxes, picked_score, picked_labels


def evaluate_with_layout(
    parsed_layout, predicate, num_parsed_layout_frames, height, width, verbose=False
):
    condition = parse.parsed_layout_to_condition(
        parsed_layout,
        tokenizer=None,
        height=height,
        width=width,
        num_parsed_layout_frames=num_parsed_layout_frames,
        num_condition_frames=num_parsed_layout_frames,
        strip_phrases=True,
        verbose=True,
    )

    print("condition:", condition)

    prompt_type = predicate.type
    success = predicate(condition, verbose=verbose)

    return prompt_type, success


def to_gen_box_format(box, width, height, rounding):
    # Input: xyxy, ranging from 0 to 1
    # Output: xywh, unnormalized (in pixels)
    x_min, y_min, x_max, y_max = box
    if rounding:
        return [
            round(x_min * width),
            round(y_min * height),
            round((x_max - x_min) * width),
            round((y_max - y_min) * height),
        ]
    return [
        x_min * width,
        y_min * height,
        (x_max - x_min) * width,
        (y_max - y_min) * height,
    ]
