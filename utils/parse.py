from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
import os
from . import guidance
from collections import namedtuple
import imageio
import shutil

Condition = namedtuple(
    "Condition", ["prompt", "boxes", "phrases", "object_positions", "token_map"]
)

img_dir = "imgs"

# h, w used in the layouts
size = (512, 512)
size_h, size_w = size
# print(f"Using box scale: {size}")


def draw_boxes(condition, frame_index=None):
    boxes, phrases = condition.boxes, condition.phrases

    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for box_ind, (box, name) in enumerate(zip(boxes, phrases)):
        if isinstance(box, dict):
            if frame_index not in box:
                continue
        else:
            if frame_index >= len(box):
                continue

        box = box[frame_index] if frame_index is not None else box
        # Each phrase may be a list to allow different phrase per timestep
        name = (
            name[frame_index]
            if frame_index is not None and isinstance(name, (dict, list, tuple))
            else name
        )

        # This ensures different frames have the same box color.
        rng = np.random.default_rng(box_ind)
        c = rng.random((1, 3)) * 0.6 + 0.4
        [bbox_x, bbox_y, bbox_x_max, bbox_y_max] = box
        if bbox_x_max <= bbox_x or bbox_y_max <= bbox_y:
            # Filters out the box in the frames without this box
            continue
        bbox_x, bbox_y, bbox_x_max, bbox_y_max = (
            bbox_x * size_w,
            bbox_y * size_h,
            bbox_x_max * size_w,
            bbox_y_max * size_h,
        )
        poly = [
            [bbox_x, bbox_y],
            [bbox_x, bbox_y_max],
            [bbox_x_max, bbox_y_max],
            [bbox_x_max, bbox_y],
        ]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(c)

        # print(ann)
        ax.text(
            bbox_x,
            bbox_y,
            name,
            style="italic",
            bbox={"facecolor": "white", "alpha": 0.7, "pad": 5},
        )

    p = PatchCollection(polygons, facecolor="none", edgecolors=color, linewidths=2)
    ax.add_collection(p)


def show_boxes(
    condition, frame_index=None, ind=None, show=True, show_prompt=True, save=False
):
    """
    This draws the boxes in `frame_index`.
    """
    boxes, phrases = condition.boxes, condition.phrases

    if len(boxes) == 0:
        return

    # White background (to allow line to show on the edge)
    I = np.ones((size[0] + 4, size[1] + 4, 3), dtype=np.uint8) * 255

    plt.imshow(I)
    plt.axis("off")

    bg_prompt = getattr(condition, "prompt", None)
    neg_prompt = getattr(condition, "neg_prompt", None)

    ax = plt.gca()
    if show_prompt and bg_prompt is not None:
        ax.text(
            0,
            0,
            bg_prompt + f"(Neg: {neg_prompt})" if neg_prompt else bg_prompt,
            style="italic",
            bbox={"facecolor": "white", "alpha": 0.7, "pad": 5},
        )
    c = np.zeros((1, 3))
    [bbox_x, bbox_y, bbox_w, bbox_h] = (0, 0, size[1], size[0])
    poly = [
        [bbox_x, bbox_y],
        [bbox_x, bbox_y + bbox_h],
        [bbox_x + bbox_w, bbox_y + bbox_h],
        [bbox_x + bbox_w, bbox_y],
    ]
    np_poly = np.array(poly).reshape((4, 2))
    polygons = [Polygon(np_poly)]
    color = [c]
    p = PatchCollection(polygons, facecolor="none", edgecolors=color, linewidths=2)
    ax.add_collection(p)

    draw_boxes(condition, frame_index=frame_index)
    if show:
        plt.show()

    if save:
        print("Saved to", f"{img_dir}/boxes.png", f"ind: {ind}")
        plt.savefig(f"{img_dir}/boxes.png")
        if ind is not None:
            shutil.copy(f"{img_dir}/boxes.png", f"{img_dir}/boxes_{ind}.png")


def show_video_boxes(
    condition,
    figsize=(4, 4),
    ind=None,
    show=False,
    save=False,
    save_each_frame=False,
    fps=8,
    save_name="boxes",
    **kwargs,
):
    boxes, phrases = condition.boxes, condition.phrases

    assert len(boxes) == len(phrases), f"{len(boxes)} != {len(phrases)}"

    if len(boxes) == 0:
        return

    num_frames = len(boxes[0])

    boxes_frames = []

    for frame_index in range(num_frames):
        fig = plt.figure(figsize=figsize)
        # https://stackoverflow.com/questions/7821518/save-plot-to-numpy-array
        show_boxes(condition, frame_index=frame_index, show=False, save=False, **kwargs)
        # If we haven't already shown or saved the plot, then we need to
        # draw the figure first...
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        boxes_frames.append(data)

    if show:
        video = imageio.mimsave(
            imageio.RETURN_BYTES,
            boxes_frames,
            format="gif",
            loop=0,
            duration=1000 * 1 / fps,
        )
        from IPython.display import display, Image as IPyImage

        display(IPyImage(data=video, format="gif"))

    if save:
        imageio.mimsave(
            f"{img_dir}/{save_name}.gif",
            boxes_frames,
            format="gif",
            loop=0,
            duration=1000 * 1 / fps,
        )
        if ind is not None:
            shutil.copy(
                f"{img_dir}/{save_name}.gif", f"{img_dir}/{save_name}_{ind}.gif"
            )
        print(f'Saved to "{img_dir}/{save_name}.gif"', f"ind: {ind}")

    if save_each_frame:
        os.makedirs(f"{img_dir}/{save_name}", exist_ok=True)
        for frame_ind, frame in enumerate(boxes_frames):
            imageio.imsave(
                f"{img_dir}/{save_name}/{frame_ind}.png", frame, format="png"
            )
        print(f'Saved frames to "{img_dir}/{save_name}"', f"ind: {ind}")


def show_masks(masks):
    masks_to_show = np.zeros((*size, 3), dtype=np.float32)
    for mask in masks:
        c = np.random.random((3,)) * 0.6 + 0.4

        masks_to_show += mask[..., None] * c[None, None, :]
    plt.imshow(masks_to_show)
    plt.savefig(f"{img_dir}/masks.png")
    plt.show()
    plt.close()


def convert_box(box, height, width):
    # box: x, y, w, h (in 512 format) -> x_min, y_min, x_max, y_max
    x_min, y_min = box[0] / width, box[1] / height
    w_box, h_box = box[2] / width, box[3] / height

    x_max, y_max = x_min + w_box, y_min + h_box

    return x_min, y_min, x_max, y_max


def interpolate_box(box, num_input_frames=6, num_output_frames=24, repeat=1):
    output_boxes = np.zeros((num_output_frames, 4))
    box_time_indices = np.sort(list(box.keys()))
    xs = np.concatenate(
        [box_time_indices / (num_input_frames - 1) + i for i in range(repeat)]
    )
    # The subtraction is to prevent the boundary effect with modulus.
    xs_query = np.linspace(0, repeat - 1e-5, num_output_frames)
    mask = np.isin(np.floor((xs_query % 1.0) * num_input_frames), box_time_indices)

    # 4: x_min, y_min, x_max, y_max
    for i in range(4):
        ys = np.array(
            [box[box_time_index][i] for box_time_index in box_time_indices] * repeat
        )
        # If the mask is False (the object does not exist in this timestep, the box has all items 0)
        output_boxes[:, i] = np.interp(xs_query, xs, ys) * mask

    return output_boxes.tolist()


def parsed_layout_to_condition(
    parsed_layout,
    height,
    width,
    num_parsed_layout_frames=6,
    num_condition_frames=24,
    interpolate_boxes=True,
    tokenizer=None,
    output_phrase_per_timestep=False,
    add_background_to_prompt=True,
    strip_phrases=False,
    verbose=False,
):
    """
    Infer condition from parsed layout.
    Boxes can appear or disappear.
    """

    prompt = parsed_layout["Prompt"]

    if add_background_to_prompt and parsed_layout["Background keyword"]:
        prompt += f", {parsed_layout['Background keyword']} background"

    id_to_phrase, id_to_box = {}, {}

    box_ids = []

    for frame_ind in range(num_parsed_layout_frames):
        object_dicts = parsed_layout[f"Frame {frame_ind + 1}"]
        for object_dict in object_dicts:
            current_box_id = object_dict["id"]
            if current_box_id not in id_to_phrase:
                if output_phrase_per_timestep:
                    # Only the phrase at the first occurrence is used if `output_phrase_per_timestep` is False
                    id_to_phrase[current_box_id] = {}
                else:
                    id_to_phrase[current_box_id] = (
                        object_dict["name"]
                        if "name" in object_dict
                        else object_dict["keyword"]
                    )

                # Use `dict` to handle appearance and disappearance of objects
                id_to_box[current_box_id] = {}

                box_ids.append(current_box_id)

            box = object_dict["box"]
            converted_box = convert_box(box, height=height, width=width)
            id_to_box[current_box_id][frame_ind] = converted_box

            if output_phrase_per_timestep:
                id_to_phrase[current_box_id][frame_ind] = (
                    object_dict["name"]
                    if "name" in object_dict
                    else object_dict["keyword"]
                )

    boxes = [id_to_box[box_id] for box_id in box_ids]
    phrases = [id_to_phrase[box_id] for box_id in box_ids]

    if verbose:
        boxes_before_interpolation = boxes

    # Frames in interpolated boxes are consecutive, but some boxes may have all coordinates as 0 to indicate disappearance
    if interpolate_boxes:
        assert (
            not output_phrase_per_timestep
        ), "box interpolation with phrase per timestep is not implemented"
        boxes = [
            interpolate_box(
                box,
                num_parsed_layout_frames,
                num_condition_frames,
                repeat=parsed_layout.get("Repeat", 1),
            )
            for box in boxes
        ]

    if tokenizer is not None:
        for phrase in phrases:
            found, _ = guidance.refine_phrase(prompt, phrase, verbose=True)

            if not found:
                # Suffix the prompt with object name (before the refinement) for attention guidance if object is not in the prompt, using "|" to separate the prompt and the suffix
                prompt += "| " + phrase

                print(f'**Adding {phrase} to the prompt. Using prompt: "{prompt}"')

        # `phrases` might not correspond to the first occurrence in the prompt, which is not handled now.
        token_map = guidance.get_token_map(
            tokenizer, prompt=prompt, verbose=verbose, padding="do_not_pad"
        )
        object_positions = guidance.get_phrase_indices(
            tokenizer, prompt, phrases, token_map=token_map, verbose=verbose
        )
    else:
        token_map = None
        object_positions = None

    if verbose:
        print("prompt:", prompt)
        print("boxes (before interpolation):", boxes_before_interpolation)
        if verbose >= 2:
            print("boxes (after interpolation):", np.round(np.array(boxes), 2))
        print("phrases:", phrases)
        if object_positions is not None:
            print("object_positions:", object_positions)

    if strip_phrases:
        phrases = [phrase.strip("1234567890 ") for phrase in phrases]

    return Condition(prompt, boxes, phrases, object_positions, token_map)
