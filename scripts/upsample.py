import argparse
import joblib
import torch
import imageio
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
from diffusers import StableDiffusionXLImg2ImgPipeline, VideoToVideoSDPipeline
from diffusers import DPMSolverMultistepScheduler


def prepare_init_upsampled(video_path, horizontal):
    video = joblib.load(video_path)

    if horizontal:
        video = [
            Image.fromarray(frame).resize((1024, 576), Image.LANCZOS) for frame in video
        ]
    else:
        video = [
            Image.fromarray(frame).resize((1024, 1024), Image.LANCZOS)
            for frame in video
        ]

    return video


def save_images_to_video(images, output_video_path, frame_rate=8.0):
    """Save a list of images to a video file."""
    if not len(images):
        print("No images to process.")
        return

    # Assuming all images are the same size, get dimensions from the first image
    height, width, layers = images[0].shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec definition
    video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    for img in images:
        bgr_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        video.write(bgr_image)

    video.release()  # Release the video writer
    print(f"Video saved to {output_video_path}")


def upsample_zsxl(
    video_path,
    prompt,
    horizontal,
    negative_prompt,
    seed,
    strength,
    use_zssdxl,
    output_mp4,
    fps=8,
):
    save_path = video_path.replace(
        ".joblib", "_zsxl" if strength == 0.35 else f"_zsxl_s{strength}"
    )
    if not os.path.exists(save_path + ".joblib"):
        video = prepare_init_upsampled(video_path, horizontal)
        g = torch.manual_seed(seed)
        video_frames_xl = pipe_xl(
            prompt,
            negative_prompt=negative_prompt,
            video=video,
            strength=strength,
            generator=g,
        ).frames[0]
        assert not os.path.exists(save_path + ".joblib"), save_path + ".joblib"
        video_frames_xl = (video_frames_xl * 255.0).astype(np.uint8)
        if output_mp4:
            save_images_to_video(
                video_frames_xl, save_path + ".mp4", frame_rate=fps + 0.01
            )
        imageio.mimsave(
            save_path + ".gif",
            video_frames_xl,
            format="gif",
            loop=0,
            duration=1000 * 1 / fps,
        )
        joblib.dump(video_frames_xl, save_path + ".joblib", compress=("bz2", 3))
        print(f"Zeroscope XL upsampled image saved at: {save_path + '.gif'}")
    else:
        print(f"{save_path + '.joblib'} exists, skipping")

    if use_zssdxl:
        upsample_sdxl(
            save_path + ".joblib",
            prompt,
            horizontal,
            negative_prompt,
            seed,
            strength=0.1,
        )


def upsample_sdxl(video_path, prompt, horizontal, negative_prompt, seed, strength):
    save_path = video_path.replace(
        ".joblib", "_sdxl" if strength == 0.35 else f"_sdxl_s{strength}"
    )
    if not os.path.exists(save_path + ".joblib"):
        video = prepare_init_upsampled(video_path, horizontal)
        pipe_sdxl.set_progress_bar_config(disable=True)
        video_frames_sdxl = []
        for video_frame in tqdm(video):
            g = torch.manual_seed(seed)
            image = pipe_sdxl(
                prompt,
                image=video_frame,
                negative_prompt=negative_prompt,
                strength=strength,
                generator=g,
            ).images[0]
            video_frames_sdxl.append(np.asarray(image))
        assert not os.path.exists(save_path + ".joblib"), save_path + ".joblib"
        imageio.mimsave(save_path + ".gif", video_frames_sdxl, format="gif", loop=0)
        joblib.dump(video_frames_sdxl, save_path + ".joblib", compress=("bz2", 3))

        print(f"SDXL upsampled image saved at: {save_path + '.gif'}")
    else:
        print(f"{save_path + '.joblib'} exists, skipping")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
        type=str,
        help="path to videos in joblib format",
    )
    parser.add_argument("--prompts", nargs="+", required=True, type=str, help="prompts")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="dull, gray, unrealistic, colorless, drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    )
    parser.add_argument("--use_zsxl", action="store_true")
    parser.add_argument("--use_sdxl", action="store_true")
    parser.add_argument("--use_zssdxl", action="store_true")
    parser.add_argument(
        "--horizontal",
        action="store_true",
        help="If True, the video is assumed to be horizontal (576x320 to 1024x576). If False, squared (512x512 to 1024x1024).",
    )
    parser.add_argument("--output-mp4", action="store_true", help="Store mp4 videos.")

    args = parser.parse_args()

    if args.use_zsxl:
        pipe_xl = VideoToVideoSDPipeline.from_pretrained(
            "cerspense/zeroscope_v2_XL", torch_dtype=torch.float16
        )
        pipe_xl.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe_xl.scheduler.config
        )
        pipe_xl.enable_model_cpu_offload()
        pipe_xl.enable_vae_slicing()

    if args.use_sdxl or args.use_zssdxl:
        pipe_sdxl = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        pipe_sdxl = pipe_sdxl.to("cuda")

    if len(args.prompts) == 1 and len(args.videos) > 1:
        args.prompts = args.prompts * len(args.videos)

    for video_path, prompt in tqdm(zip(args.videos, args.prompts)):
        video_path = video_path.replace(".gif", ".joblib")
        print(f"Video path: {video_path}, prompt: {prompt}")

        if args.use_zsxl:
            upsample_zsxl(
                video_path=video_path,
                prompt=prompt,
                horizontal=args.horizontal,
                negative_prompt=args.negative_prompt,
                seed=args.seed,
                strength=args.strength,
                use_zssdxl=args.use_zssdxl,
                output_mp4=args.output_mp4,
            )

        if args.use_sdxl:
            upsample_sdxl(
                video_path=video_path,
                prompt=prompt,
                horizontal=args.horizontal,
                negative_prompt=args.negative_prompt,
                seed=args.seed,
                strength=args.strength,
            )
