import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import guidance, schedule
import utils
from PIL import Image
import gc
import numpy as np
from .attention import GatedSelfAttentionDense
from .models import process_input_embeddings, torch_device
import warnings

DEFAULT_GUIDANCE_ATTN_KEYS = [
    ("down", 2, 0, 0),
    ("down", 2, 1, 0),
    ("up", 1, 0, 0),
    ("up", 1, 1, 0),
]


def latent_backward_guidance(
    scheduler,
    unet,
    cond_embeddings,
    index,
    bboxes,
    object_positions,
    t,
    latents,
    loss,
    loss_scale=30,
    loss_threshold=0.2,
    max_iter=5,
    max_index_step=10,
    cross_attention_kwargs=None,
    guidance_attn_keys=None,
    verbose=False,
    return_saved_attn=False,
    clear_cache=False,
    **kwargs,
):
    """
    return_saved_attn: return the saved attention for visualizations
    """

    iteration = 0

    saved_attn_to_return = None

    if index < max_index_step:
        if isinstance(max_iter, list):
            max_iter = max_iter[index]

        if verbose:
            print(
                f"time index {index}, loss: {loss.item()/loss_scale:.3f} (de-scaled with scale {loss_scale:.1f}), loss threshold: {loss_threshold:.3f}"
            )

        with torch.set_grad_enabled(True):
            while (
                loss.item() / loss_scale > loss_threshold
                and iteration < max_iter
                and index < max_index_step
            ):
                saved_attn = {}
                full_cross_attention_kwargs = {
                    "save_attn_to_dict": saved_attn,
                    "save_keys": guidance_attn_keys,
                }

                if cross_attention_kwargs is not None:
                    full_cross_attention_kwargs.update(cross_attention_kwargs)

                latents.requires_grad_(True)
                latent_model_input = latents
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=cond_embeddings,
                    cross_attention_kwargs=full_cross_attention_kwargs,
                )

                if return_saved_attn == "first":
                    if iteration == 0:
                        saved_attn_to_return = {
                            k: v.detach().cpu() for k, v in saved_attn.items()
                        }
                elif return_saved_attn == "last":
                    if iteration == max_iter - 1:
                        # It will not save if the current call returns before the last iteration
                        saved_attn_to_return = {
                            k: v.detach().cpu() for k, v in saved_attn.items()
                        }
                elif return_saved_attn:
                    raise ValueError(return_saved_attn)

                # TODO: could return the attention maps for the required blocks only and not necessarily the final output
                # update latents with guidance
                loss = (
                    guidance.compute_ca_lossv3(
                        saved_attn=saved_attn,
                        bboxes=bboxes,
                        object_positions=object_positions,
                        guidance_attn_keys=guidance_attn_keys,
                        index=index,
                        verbose=verbose,
                        **kwargs,
                    )
                    * loss_scale
                )

                if torch.isnan(loss):
                    print("**Loss is NaN**")

                del full_cross_attention_kwargs, saved_attn
                # call gc.collect() here may release some memory

                grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]

                latents.requires_grad_(False)

                if hasattr(scheduler, "alphas_cumprod"):
                    warnings.warn("Using guidance scaled with alphas_cumprod")
                    # Scaling with classifier guidance
                    alpha_prod_t = scheduler.alphas_cumprod[t]
                    # Classifier guidance: https://arxiv.org/pdf/2105.05233.pdf
                    # DDIM: https://arxiv.org/pdf/2010.02502.pdf
                    scale = (1 - alpha_prod_t) ** (0.5)

                    latents = latents - scale * grad_cond
                else:
                    # NOTE: no scaling is performed
                    warnings.warn("No scaling in guidance is performed")
                    latents = latents - grad_cond
                iteration += 1

                if clear_cache:
                    gc.collect()
                    torch.cuda.empty_cache()

                if verbose:
                    print(
                        f"time index {index}, loss: {loss.item()/loss_scale:.3f}, loss threshold: {loss_threshold:.3f}, iteration: {iteration}"
                    )

    if return_saved_attn:
        return latents, loss, saved_attn_to_return
    return latents, loss


@torch.no_grad()
def encode(model_dict, image, generator):
    """
    image should be a PIL object or numpy array with range 0 to 255
    """

    vae, dtype = model_dict.vae, model_dict.dtype

    if isinstance(image, Image.Image):
        w, h = image.size
        assert (
            w % 8 == 0 and h % 8 == 0
        ), f"h ({h}) and w ({w}) should be a multiple of 8"
        # w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8
        # image = np.array(image.resize((w, h), resample=Image.Resampling.LANCZOS))[None, :]
        image = np.array(image)

    if isinstance(image, np.ndarray):
        assert (
            image.dtype == np.uint8
        ), f"Should have dtype uint8 (dtype: {image.dtype})"
        image = image.astype(np.float32) / 255.0
        image = image[None, ...]
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)

    assert isinstance(image, torch.Tensor), f"type of image: {type(image)}"

    image = image.to(device=torch_device, dtype=dtype)
    latents = vae.encode(image).latent_dist.sample(generator)

    latents = vae.config.scaling_factor * latents

    return latents


@torch.no_grad()
def decode(vae, latents):
    # scale and decode the image latents with vae
    scaled_latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(scaled_latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")

    return images


def generate_semantic_guidance(
    model_dict,
    latents,
    input_embeddings,
    num_inference_steps,
    bboxes,
    phrases,
    object_positions,
    guidance_scale=7.5,
    semantic_guidance_kwargs=None,
    return_cross_attn=False,
    return_saved_cross_attn=False,
    saved_cross_attn_keys=None,
    return_cond_ca_only=False,
    return_token_ca_only=None,
    offload_guidance_cross_attn_to_cpu=False,
    offload_cross_attn_to_cpu=False,
    offload_latents_to_cpu=True,
    return_box_vis=False,
    show_progress=True,
    save_all_latents=False,
    dynamic_num_inference_steps=False,
    fast_after_steps=None,
    fast_rate=2,
    additional_guidance_cross_attention_kwargs={},
    custom_latent_backward_guidance=None,
):
    """
    object_positions: object indices in text tokens
    return_cross_attn: should be deprecated. Use `return_saved_cross_attn` and the new format.
    """
    vae, tokenizer, text_encoder, unet, scheduler, dtype = (
        model_dict.vae,
        model_dict.tokenizer,
        model_dict.text_encoder,
        model_dict.unet,
        model_dict.scheduler,
        model_dict.dtype,
    )
    text_embeddings, uncond_embeddings, cond_embeddings = input_embeddings

    # Just in case that we have in-place ops
    latents = latents.clone()

    if save_all_latents:
        # offload to cpu to save space
        if offload_latents_to_cpu:
            latents_all = [latents.cpu()]
        else:
            latents_all = [latents]

    scheduler.set_timesteps(num_inference_steps)
    if fast_after_steps is not None:
        scheduler.timesteps = schedule.get_fast_schedule(
            scheduler.timesteps, fast_after_steps, fast_rate
        )

    if dynamic_num_inference_steps:
        original_num_inference_steps = scheduler.num_inference_steps

    cross_attention_probs_down = []
    cross_attention_probs_mid = []
    cross_attention_probs_up = []

    loss = torch.tensor(10000.0)

    # TODO: we can also save necessary tokens only to save memory.
    # offload_guidance_cross_attn_to_cpu does not save too much since we only store attention map for each timestep.
    guidance_cross_attention_kwargs = {
        "offload_cross_attn_to_cpu": offload_guidance_cross_attn_to_cpu,
        "enable_flash_attn": False,
        **additional_guidance_cross_attention_kwargs,
    }

    if return_saved_cross_attn:
        saved_attns = []

    main_cross_attention_kwargs = {
        "offload_cross_attn_to_cpu": offload_cross_attn_to_cpu,
        "return_cond_ca_only": return_cond_ca_only,
        "return_token_ca_only": return_token_ca_only,
        "save_keys": saved_cross_attn_keys,
    }

    # Repeating keys leads to different weights for each key.
    # assert len(set(semantic_guidance_kwargs['guidance_attn_keys'])) == len(semantic_guidance_kwargs['guidance_attn_keys']), f"guidance_attn_keys not unique: {semantic_guidance_kwargs['guidance_attn_keys']}"

    for index, t in enumerate(tqdm(scheduler.timesteps, disable=not show_progress)):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.

        if bboxes:
            if custom_latent_backward_guidance:
                latents, loss = custom_latent_backward_guidance(
                    scheduler,
                    unet,
                    cond_embeddings,
                    index,
                    bboxes,
                    object_positions,
                    t,
                    latents,
                    loss,
                    cross_attention_kwargs=guidance_cross_attention_kwargs,
                    **semantic_guidance_kwargs,
                )
            else:
                # If encountered None in `guidance_attn_keys`, please be sure to check whether `guidance_attn_keys` is added in `semantic_guidance_kwargs`. Default value has been removed.
                latents, loss = latent_backward_guidance(
                    scheduler,
                    unet,
                    cond_embeddings,
                    index,
                    bboxes,
                    object_positions,
                    t,
                    latents,
                    loss,
                    cross_attention_kwargs=guidance_cross_attention_kwargs,
                    **semantic_guidance_kwargs,
                )

        # predict the noise residual
        with torch.no_grad():
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(
                latent_model_input, timestep=t
            )

            main_cross_attention_kwargs["save_attn_to_dict"] = {}

            unet_output = unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                return_cross_attention_probs=return_cross_attn,
                cross_attention_kwargs=main_cross_attention_kwargs,
            )
            noise_pred = unet_output.sample

            if return_cross_attn:
                cross_attention_probs_down.append(
                    unet_output.cross_attention_probs_down
                )
                cross_attention_probs_mid.append(unet_output.cross_attention_probs_mid)
                cross_attention_probs_up.append(unet_output.cross_attention_probs_up)

            if return_saved_cross_attn:
                saved_attns.append(main_cross_attention_kwargs["save_attn_to_dict"])

                del main_cross_attention_kwargs["save_attn_to_dict"]

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        if dynamic_num_inference_steps:
            schedule.dynamically_adjust_inference_steps(scheduler, index, t)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        if save_all_latents:
            if offload_latents_to_cpu:
                latents_all.append(latents.cpu())
            else:
                latents_all.append(latents)

    if dynamic_num_inference_steps:
        # Restore num_inference_steps to avoid confusion in the next generation if it is not dynamic
        scheduler.num_inference_steps = original_num_inference_steps

    images = decode(vae, latents)

    ret = [latents, images]

    if return_cross_attn:
        ret.append(
            (
                cross_attention_probs_down,
                cross_attention_probs_mid,
                cross_attention_probs_up,
            )
        )
    if return_saved_cross_attn:
        ret.append(saved_attns)
    if return_box_vis:
        pil_images = [
            utils.draw_box(Image.fromarray(image), bboxes, phrases) for image in images
        ]
        ret.append(pil_images)
    if save_all_latents:
        latents_all = torch.stack(latents_all, dim=0)
        ret.append(latents_all)
    return tuple(ret)
