import torch
from PIL import ImageDraw
import numpy as np
import os
import gc
import math
from typing import List
import cv2
import skvideo.io

torch_device = "cuda"


def draw_box(pil_img, bboxes, phrases, ignore_all_zeros=True):
    W, H = pil_img.size
    draw = ImageDraw.Draw(pil_img)

    for obj_bbox, phrase in zip(bboxes, phrases):
        x_0, y_0, x_1, y_1 = obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3]
        if ignore_all_zeros and x_0 == 0 and y_0 == 0 and x_1 == 0 and y_1 == 0:
            continue
        draw.rectangle(
            [int(x_0 * W), int(y_0 * H), int(x_1 * W), int(y_1 * H)],
            outline="red",
            width=5,
        )
        draw.text(
            (int(x_0 * W) + 5, int(y_0 * H) + 5), phrase, font=None, fill=(255, 0, 0)
        )

    return pil_img


def get_centered_box(
    box,
    horizontal_center_only=True,
    vertical_placement="centered",
    vertical_center=0.5,
    floor_padding=None,
):
    x_min, y_min, x_max, y_max = box
    w = x_max - x_min

    x_min_new = 0.5 - w / 2
    x_max_new = 0.5 + w / 2

    if horizontal_center_only:
        return [x_min_new, y_min, x_max_new, y_max]

    h = y_max - y_min

    if vertical_placement == "centered":
        assert (
            floor_padding is None
        ), "Set vertical_placement to floor_padding to use floor padding"

        y_min_new = vertical_center - h / 2
        y_max_new = vertical_center + h / 2
    elif vertical_placement == "floor_padding":
        # Ignores `vertical_center`

        y_max_new = 1 - floor_padding
        y_min_new = y_max_new - h
    else:
        raise ValueError(f"Unknown vertical placement: {vertical_placement}")

    return [x_min_new, y_min_new, x_max_new, y_max_new]


# NOTE: this changes the behavior of the function
def proportion_to_mask(obj_box, H, W, use_legacy=False, return_np=False):
    x_min, y_min, x_max, y_max = scale_proportion(obj_box, H, W, use_legacy)
    if return_np:
        mask = np.zeros((H, W))
    else:
        mask = torch.zeros(H, W).to(torch_device)
    mask[y_min:y_max, x_min:x_max] = 1.0

    return mask


def scale_proportion(obj_box, H, W, use_legacy=False):
    if use_legacy:
        # Bias towards the top-left corner
        x_min, y_min, x_max, y_max = (
            int(obj_box[0] * W),
            int(obj_box[1] * H),
            int(obj_box[2] * W),
            int(obj_box[3] * H),
        )
    else:
        # Separately rounding box_w and box_h to allow shift invariant box sizes. Otherwise box sizes may change when both coordinates being rounded end with ".5".
        x_min, y_min = round(obj_box[0] * W), round(obj_box[1] * H)
        box_w, box_h = (
            round((obj_box[2] - obj_box[0]) * W),
            round((obj_box[3] - obj_box[1]) * H),
        )
        x_max, y_max = x_min + box_w, y_min + box_h

        x_min, y_min = max(x_min, 0), max(y_min, 0)
        x_max, y_max = min(x_max, W), min(y_max, H)

    return x_min, y_min, x_max, y_max


def binary_mask_to_box(mask, enlarge_box_by_one=True, w_scale=1, h_scale=1):
    if isinstance(mask, torch.Tensor):
        mask_loc = torch.where(mask)
    else:
        mask_loc = np.where(mask)
    height, width = mask.shape
    if len(mask_loc) == 0:
        raise ValueError("The mask is empty")
    if enlarge_box_by_one:
        ymin, ymax = max(min(mask_loc[0]) - 1, 0), min(max(mask_loc[0]) + 1, height)
        xmin, xmax = max(min(mask_loc[1]) - 1, 0), min(max(mask_loc[1]) + 1, width)
    else:
        ymin, ymax = min(mask_loc[0]), max(mask_loc[0])
        xmin, xmax = min(mask_loc[1]), max(mask_loc[1])
    box = [xmin * w_scale, ymin * h_scale, xmax * w_scale, ymax * h_scale]

    return box


def binary_mask_to_box_mask(mask, to_device=True):
    box = binary_mask_to_box(mask)
    x_min, y_min, x_max, y_max = box

    H, W = mask.shape
    mask = torch.zeros(H, W)
    if to_device:
        mask = mask.to(torch_device)
    mask[y_min : y_max + 1, x_min : x_max + 1] = 1.0

    return mask


def binary_mask_to_center(mask, normalize=False):
    """
    This computes the mass center of the mask.
    normalize: the coords range from 0 to 1

    Reference: https://stackoverflow.com/a/66184125
    """
    h, w = mask.shape

    total = mask.sum()
    if isinstance(mask, torch.Tensor):
        x_coord = ((mask.sum(dim=0) @ torch.arange(w)) / total).item()
        y_coord = ((mask.sum(dim=1) @ torch.arange(h)) / total).item()
    else:
        x_coord = (mask.sum(axis=0) @ np.arange(w)) / total
        y_coord = (mask.sum(axis=1) @ np.arange(h)) / total

    if normalize:
        x_coord, y_coord = x_coord / w, y_coord / h
    return x_coord, y_coord


def iou(mask, masks, eps=1e-6):
    # mask: [h, w], masks: [n, h, w]
    mask = mask[None].astype(bool)
    masks = masks.astype(bool)
    i = (mask & masks).sum(axis=(1, 2))
    u = (mask | masks).sum(axis=(1, 2))

    return i / (u + eps)


def free_memory():
    gc.collect()
    torch.cuda.empty_cache()


def expand_overall_bboxes(overall_bboxes):
    """
    Expand overall bboxes from a 3d list to 2d list:
    Input: [[box 1 for phrase 1, box 2 for phrase 1], ...]
    Output: [box 1, box 2, ...]
    """
    return sum(overall_bboxes, start=[])


def shift_tensor(
    tensor,
    x_offset,
    y_offset,
    base_w=8,
    base_h=8,
    offset_normalized=False,
    ignore_last_dim=False,
):
    """base_w and base_h: make sure the shift is aligned in the latent and multiple levels of cross attention"""
    if ignore_last_dim:
        tensor_h, tensor_w = tensor.shape[-3:-1]
    else:
        tensor_h, tensor_w = tensor.shape[-2:]
    if offset_normalized:
        assert (
            tensor_h % base_h == 0 and tensor_w % base_w == 0
        ), f"{tensor_h, tensor_w} is not a multiple of {base_h, base_w}"
        scale_from_base_h, scale_from_base_w = tensor_h // base_h, tensor_w // base_w
        x_offset, y_offset = (
            round(x_offset * base_w) * scale_from_base_w,
            round(y_offset * base_h) * scale_from_base_h,
        )
    new_tensor = torch.zeros_like(tensor)

    overlap_w = tensor_w - abs(x_offset)
    overlap_h = tensor_h - abs(y_offset)

    if y_offset >= 0:
        y_src_start = 0
        y_dest_start = y_offset
    else:
        y_src_start = -y_offset
        y_dest_start = 0

    if x_offset >= 0:
        x_src_start = 0
        x_dest_start = x_offset
    else:
        x_src_start = -x_offset
        x_dest_start = 0

    if ignore_last_dim:
        # For cross attention maps, the third to last and the second to last are the 2D dimensions after unflatten.
        new_tensor[
            ...,
            y_dest_start : y_dest_start + overlap_h,
            x_dest_start : x_dest_start + overlap_w,
            :,
        ] = tensor[
            ...,
            y_src_start : y_src_start + overlap_h,
            x_src_start : x_src_start + overlap_w,
            :,
        ]
    else:
        new_tensor[
            ...,
            y_dest_start : y_dest_start + overlap_h,
            x_dest_start : x_dest_start + overlap_w,
        ] = tensor[
            ...,
            y_src_start : y_src_start + overlap_h,
            x_src_start : x_src_start + overlap_w,
        ]

    return new_tensor


def get_hw_from_attn_dim(attn_dim, base_attn_dim):
    # base_attn_dim: (40, 72) for zeroscope (width 576, height 320)
    scale = int(math.sqrt((base_attn_dim[0] * base_attn_dim[1]) / attn_dim))
    return base_attn_dim[0] // scale, base_attn_dim[1] // scale


# Reference: https://github.com/huggingface/diffusers/blob/v0.20.0/src/diffusers/utils/testing_utils.py#L400
def export_to_video(
    video_frames: List[np.ndarray],
    output_video_path: str,
    fps: int = 8,
    fourcc: str = "mp4v",
    use_opencv=False,
    crf=17,
) -> str:
    if use_opencv:
        # This requires a cv2 installation that has video encoder support.

        fourcc = cv2.VideoWriter_fourcc(*fourcc)
        h, w, c = video_frames[0].shape
        video_writer = cv2.VideoWriter(
            output_video_path, fourcc, fps=fps, frameSize=(w, h)
        )
        for i in range(len(video_frames)):
            img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
            video_writer.write(img)
    else:
        skvideo.io.vwrite(
            output_video_path,
            video_frames,
            inputdict={"-framerate": str(fps)},
            outputdict={"-vcodec": "libx264", "-pix_fmt": "yuv420p", "-crf": str(crf)},
        )
    return output_video_path


def multiline_input(prompt, return_on_empty_lines=True):
    # Adapted from https://stackoverflow.com/questions/30239092/how-to-get-multiline-input-from-the-user

    print(prompt, end="", flush=True)
    contents = ""
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "" and return_on_empty_lines:
            break
        contents += line + "\n"

    return contents


def find_gen_dir(gen_name, create_dir=True):
    base_save_dir = f"img_generations/{gen_name}"
    run_ind = 0

    while True:
        gen_dir = f"{base_save_dir}/run{run_ind}"
        if not os.path.exists(gen_dir):
            break
        run_ind += 1

    print(f"Save results at {gen_dir}")
    if create_dir:
        os.makedirs(gen_dir, exist_ok=False)

    return gen_dir
