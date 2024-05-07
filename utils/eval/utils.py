import numpy as np
import inflect
import re
from utils.parse import Condition

p = inflect.engine()

def find_word_after(text, word):
    pattern = r"\b" + re.escape(word) + r"\s+(.+)"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return None


word_to_num_mapping = {p.number_to_words(i): i for i in range(1, 21)}

locations_xyxy = {
    ("left", "right"): (lambda box1, box2: (box1[0] + box1[2]) < (box2[0] + box2[2])),
    ("right", "left"): (lambda box1, box2: (box1[0] + box1[2]) > (box2[0] + box2[2])),
    ("top", "bottom"): (lambda box1, box2: (box1[1] + box1[3]) < (box2[1] + box2[3])),
    ("bottom", "top"): (lambda box1, box2: (box1[1] + box1[3]) > (box2[1] + box2[3])),
}

locations_xywh = {
    ("left", "right"): (
        lambda box1, box2: box1[0] + box1[2] / 2 < box2[0] + box2[2] / 2
    ),
    ("right", "left"): (
        lambda box1, box2: box1[0] + box1[2] / 2 > box2[0] + box2[2] / 2
    ),
    ("top", "bottom"): (
        lambda box1, box2: box1[1] + box1[3] / 2 < box2[1] + box2[3] / 2
    ),
    ("bottom", "top"): (
        lambda box1, box2: box1[1] + box1[3] / 2 > box2[1] + box2[3] / 2
    ),
}


def singular(noun):
    singular_noun = p.singular_noun(noun)
    if singular_noun is False:
        return noun
    return singular_noun


def get_box(condition: Condition, name_include):
    # This prevents substring match on non-word boundaries: carrot vs car
    box_match = [
        any(
            [
                (
                    (name_include_item + " ") in phrase
                    or phrase.endswith(name_include_item)
                )
                for name_include_item in name_include
            ]
        )
        for phrase in condition.phrases
    ]

    if not any(box_match):
        return None

    boxes = condition.boxes
    box_ind = np.min(np.where(box_match)[0])

    return boxes[box_ind]


def get_box_counts(condition):
    if len(condition.boxes) == 0:
        # No boxes
        return None

    box_counts = None

    for i, box in enumerate(condition.boxes):
        if i == 0:
            num_frames = len(box)
            box_counts = [0 for _ in range(num_frames)]
        else:
            assert num_frames == len(box), f"{num_frames} != {len(box)}"
        valid_frames = box_to_valid_frames(box)

        for frame_index, valid in enumerate(valid_frames):
            if valid:
                box_counts[frame_index] += 1

    return box_counts


def predicate_numeracy(query_names, intended_count, condition, verbose=False):
    assert len(query_names) == 1
    name_include = query_names
    box_match = [
        any(
            [
                (
                    (name_include_item + " ") in phrase
                    or phrase.endswith(name_include_item)
                )
                for name_include_item in name_include
            ]
        )
        for phrase in condition.phrases
    ]

    # We do not have tracking in stage 2 evaluation, so let's put this assertion to be safe.
    # This could only be a problem for stage 1 where additional non-relevant boxes are generated, but so far we did not see additional boxes in stage 1.
    assert len(box_match) == len(
        condition.boxes
    ), "Currently do not support the case where other boxes are also generated"

    box_counts = get_box_counts(condition)

    if box_counts is None:
        majority_box_counts = 0
    else:
        majority_box_counts = np.bincount(box_counts).argmax()

    object_count = majority_box_counts
    if verbose:
        print(
            f"box_counts: {box_counts}, object_count: {object_count}, intended_count: {intended_count} (condition: {condition}, query_names: {query_names})"
        )

    success = object_count == intended_count

    return success


def box_to_valid_frames(object_box):
    object_box = np.array(object_box)
    x, y, w, h = object_box[:, 0], object_box[:, 1], object_box[:, 2], object_box[:, 3]
    # If the box has 0 width or height, it is not valid.
    valid_frames = (w != 0) & (h != 0)

    return valid_frames


def predicate_visibility(query_names, test_appearance, condition, verbose=False):
    # condition: dict with keys 'name' and 'bounding_box'

    object_box = get_box(condition, query_names)
    if not object_box:
        return False

    valid_frames = box_to_valid_frames(object_box)

    num_frames = len(valid_frames)
    first_half_index = num_frames // 2

    # Ignore the two frames in the middle since there may be discrepancies between the LLM's understanding of the middle frame and the middle frame after interpolation (in generation) and sampling (in evaluation).
    valid_frames_first_half, valid_frames_second_half = (
        valid_frames[: first_half_index - 1],
        valid_frames[first_half_index + 1 :],
    )
    present_in_first_half, present_in_second_half = (
        any(valid_frames_first_half),
        any(valid_frames_second_half),
    )

    if test_appearance:
        # Test appearing: we ensure the object is not in the first half but needs to be present in the second half.
        success = (not present_in_first_half) and present_in_second_half
    else:
        # Test disappearing: we ensure the object is in the first half but needs to be absent in the second half.
        success = present_in_first_half and (not present_in_second_half)

    if verbose:
        print(
            f"Test appearance: {test_appearance}, valid_frames: {valid_frames}, appeared at first half: {present_in_first_half}, appeared at second half: {present_in_second_half}"
        )

    return success


def predicate_attribution(
    query_names1,
    query_names2,
    modifier1,
    modifier2,
    intended_count1,
    intended_count2,
    condition,
    verbose=False,
):
    # Attribution does not use count now
    assert intended_count1 == 1 and intended_count2 == 1

    if modifier1:
        query_names1 = [f"{modifier1} {item}" for item in query_names1]
    object_box1 = get_box(condition, name_include=query_names1)

    if object_box1 is None:
        return False

    valid_frames1 = box_to_valid_frames(object_box1)
    if valid_frames1.mean() < 0.5:
        # Not detected at more than half of frames
        return False

    if query_names2 is None:
        # Only one object
        return True

    if modifier2:
        query_names2 = [f"{modifier2} {item}" for item in query_names2]
    object_box2 = get_box(condition, name_include=query_names2)

    if object_box2 is None:
        return False

    valid_frames2 = box_to_valid_frames(object_box2)
    if valid_frames2.mean() < 0.5:
        # Not detected at more than half of frames
        return False

    if verbose:
        print(f"Object box 1: {object_box1}, Object box 2: {object_box2}")

    return True


def predicate_1obj_dynamic_spatial(query_names, verify_fn, condition, verbose=False):
    object_box = get_box(condition, query_names)
    if not object_box:
        return False

    valid_frames = box_to_valid_frames(object_box)
    if not valid_frames[0] or not valid_frames[-1]:
        return False

    # For example, from the left to the right: object in the first frame is on the left compared to the object in the last frame
    success = verify_fn(object_box[0], object_box[-1])

    return success


def predicate_2obj_dynamic_spatial(
    query_names1, query_names2, verify_fn, condition, verbose=False
):
    object_box1 = get_box(condition, query_names1)
    object_box2 = get_box(condition, query_names2)

    if verbose:
        print(f"object_box1: {object_box1}, object_box2: {object_box2}")

    if not object_box1 or not object_box2:
        return False

    valid_frames1 = box_to_valid_frames(object_box1)
    valid_frames2 = box_to_valid_frames(object_box2)
    if (
        not valid_frames1[0]
        or not valid_frames2[0]
        or not valid_frames1[-1]
        or not valid_frames2[-1]
    ):
        return False

    # For example, `object 1` moving from the left of `object 2` to the right: object 1 in the first frame is on the left compared to object 1 in the first frame; object 1 in the last frame is on the right compared to object 2 in the last frame
    success1 = verify_fn(object_box1[0], object_box2[0])
    success2 = verify_fn(object_box2[-1], object_box1[-1])
    success = success1 and success2

    return success


def predicate_sequentialv2(
    query_names, verify_fn1, verify_fn2, verify_fn3, condition, verbose=False
):
    # condition: dict with keys 'name' and 'bounding_box'

    object_box = get_box(condition, query_names)
    if verbose:
        print(f"object_box: {object_box}")

    if not object_box:
        return False

    valid_frames = box_to_valid_frames(object_box)
    if verbose:
        print(f"valid_frames: {valid_frames}")

    num_frames = len(valid_frames)
    middle_frame_index = num_frames // 2

    # Need to be present in the first, the middle, and the last frame
    if (
        not valid_frames[0]
        or not valid_frames[middle_frame_index]
        or not valid_frames[-1]
    ):
        return False

    # Need to be on the right place in the first, middle, and last frame
    success1 = verify_fn1(object_box[0])
    success2 = verify_fn2(object_box[middle_frame_index])
    success3 = verify_fn3(object_box[-1])

    if verbose:
        print(
            f"success1: {success1} ({object_box[0]}), success2: {success2} ({object_box[middle_frame_index]}), success3: {success3} ({object_box[-1]})"
        )

    success = success1 and success2 and success3
    return success
