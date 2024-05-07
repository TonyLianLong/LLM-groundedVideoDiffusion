from diffusers import TextToVideoSDPipeline
from diffusers import DPMSolverMultistepScheduler
from utils import parse, vis
from prompt import negative_prompt
import torch
import numpy as np
import os

version = "modelscope"

# %%
model_key = "damo-vilab/text-to-video-ms-1.7b"

pipe = TextToVideoSDPipeline.from_pretrained(model_key, torch_dtype=torch.float16)
# The default one is DDIMScheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.unet = UNet3DConditionModel.from_pretrained(
#     model_key, subfolder="unet").to(torch.float16)
pipe.to("cuda")
pipe.enable_vae_slicing()
# No auxiliary guidance
pipe.guidance_models = None

# %%
H, W = None, None


def init(option):
    global H, W
    if option == "":
        H, W = 512, 512
    elif option == "256":
        H, W = 256, 256
    else:
        raise ValueError(f"Unknown option: {option}")

    return H, W


def run(
    parsed_layout,
    seed,
    *,
    num_inference_steps=40,
    num_frames=16,
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

    # %%
    vis.save_frames(
        f"{parse.img_dir}/video_{save_suffix}", video_frames, formats=save_formats
    )
