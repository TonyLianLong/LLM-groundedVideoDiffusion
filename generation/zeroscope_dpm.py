from diffusers import TextToVideoSDPipeline, VideoToVideoSDPipeline
from diffusers import DPMSolverMultistepScheduler
from utils import parse, vis
from prompt import negative_prompt
import torch
from PIL import Image
import numpy as np
import os

version = "zeroscope"

# %%
H, W = None, None

pipe = TextToVideoSDPipeline.from_pretrained(
    "cerspense/zeroscope_v2_576w",
    # unet = UNet3DConditionModel.from_pretrained(
    #     "cerspense/zeroscope_v2_576w", subfolder="unet"
    # ).to(torch.float16),
    torch_dtype=torch.float16,
)
# The default one is DDIMScheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")
pipe.enable_vae_slicing()
pipe_xl = None


def init(option):
    global pipe_xl, H, W

    if option == "":
        H, W = 320, 576
    elif option == "xl":
        # the base model is still in 320, 576. The xl model outputs (576, 1024).
        H, W = 320, 576

        pipe_xl = VideoToVideoSDPipeline.from_pretrained(
            "cerspense/zeroscope_v2_XL", torch_dtype=torch.float16
        )
        pipe_xl.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )
        # pipe_xl.enable_model_cpu_offload()
        pipe_xl.to("cuda")
        pipe_xl.enable_vae_slicing()
    else:
        raise ValueError(f"Unknown option: {option}")
    # WIP
    return H, W


# %%
def run(
    parsed_layout,
    seed,
    *,
    num_inference_steps=40,
    num_frames=24,
    repeat_ind=None,
    save_formats=["gif", "joblib"],
):
    prompt = parsed_layout["Prompt"]

    if repeat_ind is not None:
        save_suffix = repeat_ind
    else:
        save_suffix = f"seed{seed}"

    save_path = f"{parse.img_dir}/video_{save_suffix}.gif"
    if os.path.exists(save_path):
        print(f"Skipping {save_path}")
        return

    print("Generating")
    generator = torch.Generator(device="cuda").manual_seed(seed)

    video_frames = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        height=H,
        width=W,
        num_frames=num_frames,
        cross_attention_kwargs=None,
        generator=generator,
    ).frames
    video_frames = (video_frames[0] * 255.0).astype(np.uint8)

    if pipe_xl is not None:
        print("Refining")
        video = [Image.fromarray(frame).resize((1024, 576)) for frame in video_frames]
        video_frames_xl = pipe_xl(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            video=video,
            strength=0.6,
            generator=generator,
        ).frames

        video_frames = (video_frames[0] * 255.0).astype(np.uint8)

        print("Saving")
        vis.save_frames(
            f"{parse.img_dir}/video_xl_{save_suffix}",
            video_frames_xl,
            formats=save_formats,
        )
    else:
        vis.save_frames(
            f"{parse.img_dir}/video_{save_suffix}", video_frames, formats=save_formats
        )
