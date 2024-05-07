from models.controllable_pipeline_text_to_video_synth import TextToVideoSDPipeline
from diffusers import DPMSolverMultistepScheduler
from models.unet_3d_condition import UNet3DConditionModel
from utils import parse, vis
from prompt import negative_prompt
import utils
import numpy as np
import torch
from PIL import Image
import os

version = "lvd-gligen"

# %%
# H, W are generation H and W. box_W and box_W are for scaling the boxes to [0, 1].
pipe, H, W, box_H, box_W = None, None, None, None, None


def init(base_model):
    global pipe, H, W, box_H, box_W
    if base_model == "modelscope256":
        model_key = "longlian/text-to-video-lvd-ms"
        H, W = 256, 256
        box_H, box_W = parse.size
    elif base_model == "zeroscope":
        model_key = "longlian/text-to-video-lvd-zs"
        H, W = 320, 576
        box_H, box_W = parse.size
    else:
        raise ValueError(f"Unknown base model: {base_model}")

    pipe = TextToVideoSDPipeline.from_pretrained(
        model_key, trust_remote_code=True, torch_dtype=torch.float16
    )
    # The default one is DDIMScheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.enable_vae_slicing()

    # No auxiliary guidance
    pipe.guidance_models = None

    return H, W


# %%
upsample_scale, upsample_mode = 1, "bilinear"

# %%

# Seems like `enable_model_cpu_offload` performs deepcopy so `save_attn_to_dict` does not save the attn
cross_attention_kwargs = {
    # This is for visualizations
    # 'offload_cross_attn_to_cpu': True
}


# %%
def run(
    parsed_layout,
    seed,
    num_inference_steps=40,
    num_frames=16,
    gligen_scheduled_sampling_beta=1.0,
    repeat_ind=None,
    save_annotated_videos=False,
    save_formats=["gif", "joblib"],
):
    condition = parse.parsed_layout_to_condition(
        parsed_layout,
        tokenizer=pipe.tokenizer,
        height=box_H,
        width=box_W,
        num_condition_frames=num_frames,
        verbose=True,
    )
    prompt, bboxes, phrases, object_positions, token_map = (
        condition.prompt,
        condition.boxes,
        condition.phrases,
        condition.object_positions,
        condition.token_map,
    )

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

    lvd_gligen_boxes = []
    lvd_gligen_phrases = []
    for i in range(num_frames):
        lvd_gligen_boxes.append(
            [
                bboxes_item[i]
                for phrase, bboxes_item in zip(phrases, bboxes)
                if bboxes_item[i] != [0.0, 0.0, 0.0, 0.0]
            ]
        )
        lvd_gligen_phrases.append(
            [
                phrase
                for phrase, bboxes_item in zip(phrases, bboxes)
                if bboxes_item[i] != [0.0, 0.0, 0.0, 0.0]
            ]
        )

    video_frames = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        height=H,
        width=W,
        num_frames=num_frames,
        cross_attention_kwargs=cross_attention_kwargs,
        generator=generator,
        lvd_gligen_scheduled_sampling_beta=gligen_scheduled_sampling_beta,
        lvd_gligen_boxes=lvd_gligen_boxes,
        lvd_gligen_phrases=lvd_gligen_phrases,
    ).frames
    # `diffusers` has a backward-breaking change
    # video_frames = (video_frames[0] * 255.).astype(np.uint8)

    # %%

    if save_annotated_videos:
        annotated_frames = [
            np.array(
                utils.draw_box(
                    Image.fromarray(video_frame), [bbox[i] for bbox in bboxes], phrases
                )
            )
            for i, video_frame in enumerate(video_frames)
        ]
        vis.save_frames(
            f"{save_path}/video_seed{seed}_with_box",
            frames=annotated_frames,
            formats="gif",
        )

    vis.save_frames(
        f"{parse.img_dir}/video_{save_suffix}", video_frames, formats=save_formats
    )
