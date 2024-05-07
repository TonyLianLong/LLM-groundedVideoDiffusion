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

version = "lvd-plus"

# %%
# H, W are generation H and W. box_W and box_W are for scaling the boxes to [0, 1].
pipe, base_attn_dim, H, W, box_H, box_W = None, None, None, None, None, None


def init(base_model):
    global pipe, base_attn_dim, H, W, box_H, box_W
    if base_model == "modelscope256":
        model_key = "longlian/text-to-video-lvd-ms"
        base_attn_dim = (32, 32)
        H, W = 256, 256
        box_H, box_W = parse.size
    else:
        raise ValueError(f"Unknown base model: {base_model}")

    unet = UNet3DConditionModel.from_pretrained(
        model_key, subfolder="unet", revision="weights_only"
    ).to(torch.float16)
    pipe = TextToVideoSDPipeline.from_pretrained(
        model_key, unet=unet, torch_dtype=torch.float16, revision="weights_only"
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
# For visualizations: set to False to not save
return_guidance_saved_attn = False
# This is the main attn, not the attn for guidance.
save_keys = []

# %%
overall_guidance_attn_keys = [
    ("down", 1, 0, 0),
    ("down", 2, 0, 0),
    ("down", 2, 1, 0),
    ("up", 1, 0, 0),
    ("up", 1, 1, 0),
    ("up", 2, 2, 0),
]

# Seems like `enable_model_cpu_offload` performs deepcopy so `save_attn_to_dict` does not save the attn
cross_attention_kwargs = {
    "save_attn_to_dict": {},
    "save_keys": save_keys,
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
    loss_scale=5.0,
    loss_threshold=200.0,
    max_iter=5,
    max_index_step=10,
    fg_top_p=0.75,
    bg_top_p=0.75,
    fg_weight=1.0,
    bg_weight=4.0,
    attn_sync_weight=0.0,
    boxdiff_loss_scale=0.0,
    boxdiff_normed=True,
    com_loss_scale=0.0,
    use_ratio_based_loss=False,
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

    backward_guidance_kwargs = dict(
        bboxes=bboxes,
        object_positions=object_positions,
        loss_scale=loss_scale,
        loss_threshold=loss_threshold,
        max_iter=max_iter,
        max_index_step=max_index_step,
        fg_top_p=fg_top_p,
        bg_top_p=bg_top_p,
        fg_weight=fg_weight,
        bg_weight=bg_weight,
        use_ratio_based_loss=use_ratio_based_loss,
        guidance_attn_keys=overall_guidance_attn_keys,
        exclude_bg_heads=False,
        upsample_scale=upsample_scale,
        upsample_mode=upsample_mode,
        base_attn_dim=base_attn_dim,
        attn_sync_weight=attn_sync_weight,
        boxdiff_loss_scale=boxdiff_loss_scale,
        boxdiff_normed=boxdiff_normed,
        com_loss_scale=com_loss_scale,
        verbose=True,
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

    # print(bboxes, phrases)

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
        gligen_scheduled_sampling_beta=gligen_scheduled_sampling_beta,
        gligen_boxes=lvd_gligen_boxes,
        gligen_phrases=lvd_gligen_phrases,
        guidance_callback=None,
        backward_guidance_kwargs=backward_guidance_kwargs,
        return_guidance_saved_attn=return_guidance_saved_attn,
        guidance_type="main",
    ).frames
    video_frames = (video_frames[0] * 255.0).astype(np.uint8)

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
