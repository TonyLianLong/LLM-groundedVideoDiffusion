import matplotlib.pyplot as plt
import numpy as np
import utils
from . import parse
import imageio
import joblib


def visualize(image, title, colorbar=False, show_plot=True, **kwargs):
    plt.title(title)
    plt.imshow(image, **kwargs)
    if colorbar:
        plt.colorbar()
    if show_plot:
        plt.show()


def visualize_arrays(
    image_title_pairs,
    colorbar_index=-1,
    show_plot=True,
    figsize=None,
    no_axis=False,
    **kwargs,
):
    if figsize is not None:
        plt.figure(figsize=figsize)
    num_subplots = len(image_title_pairs)
    for idx, image_title_pair in enumerate(image_title_pairs):
        plt.subplot(1, num_subplots, idx + 1)
        if isinstance(image_title_pair, (list, tuple)):
            image, title = image_title_pair
        else:
            image, title = image_title_pair, None

        if title is not None:
            plt.title(title)

        plt.imshow(image, **kwargs)
        if no_axis:
            plt.axis("off")
        if idx == colorbar_index:
            plt.colorbar()

    if show_plot:
        plt.show()


def visualize_masked_latents(
    latents_all, masked_latents, timestep_T=False, timestep_0=True
):
    if timestep_T:
        # from T to 0
        latent_idx = 0

        plt.subplot(1, 2, 1)
        plt.title("latents_all (t=T)")
        plt.imshow(
            (
                latents_all[latent_idx, 0, :3]
                .cpu()
                .permute(1, 2, 0)
                .numpy()
                .astype(float)
                / 1.5
            ).clip(0.0, 1.0),
            cmap="gray",
        )

        plt.subplot(1, 2, 2)
        plt.title("mask latents (t=T)")
        plt.imshow(
            (
                masked_latents[latent_idx, 0, :3]
                .cpu()
                .permute(1, 2, 0)
                .numpy()
                .astype(float)
                / 1.5
            ).clip(0.0, 1.0),
            cmap="gray",
        )

        plt.show()

    if timestep_0:
        latent_idx = -1
        plt.subplot(1, 2, 1)
        plt.title("latents_all (t=0)")
        plt.imshow(
            (
                latents_all[latent_idx, 0, :3]
                .cpu()
                .permute(1, 2, 0)
                .numpy()
                .astype(float)
                / 1.5
            ).clip(0.0, 1.0),
            cmap="gray",
        )

        plt.subplot(1, 2, 2)
        plt.title("mask latents (t=0)")
        plt.imshow(
            (
                masked_latents[latent_idx, 0, :3]
                .cpu()
                .permute(1, 2, 0)
                .numpy()
                .astype(float)
                / 1.5
            ).clip(0.0, 1.0),
            cmap="gray",
        )

        plt.show()


def visualize_bboxes(bboxes, H, W):
    num_boxes = len(bboxes)
    for ind, bbox in enumerate(bboxes):
        plt.subplot(1, num_boxes, ind + 1)
        fg_mask = utils.proportion_to_mask(bbox, H, W)
        plt.title(f"transformed bbox ({ind})")
        plt.imshow(fg_mask.cpu().numpy())
    plt.show()


def save_image(image, save_prefix="", ind=None):
    global save_ind
    if save_prefix != "":
        save_prefix = save_prefix + "_"
    ind = f"{ind}_" if ind is not None else ""
    path = f"{parse.img_dir}/{save_prefix}{ind}{save_ind}.png"

    print(f"Saved to {path}")

    image.save(path)
    save_ind = save_ind + 1


def save_frames(path, frames, formats="gif", fps=8):
    if isinstance(formats, (list, tuple)):
        for format in formats:
            save_frames(path, frames, format, fps)
        return

    if formats == "gif":
        imageio.mimsave(
            f"{path}.gif", frames, format="gif", loop=0, duration=1000 * 1 / fps
        )
    elif formats == "mp4":
        utils.export_to_video(
            video_frames=frames, output_video_path=f"{path}.mp4", fps=fps
        )
    elif formats == "npz":
        np.savez_compressed(f"{path}.npz", frames)
    elif formats == "joblib":
        joblib.dump(frames, f"{path}.joblib", compress=("bz2", 3))
    else:
        raise ValueError(f"Unknown format: {formats}")
