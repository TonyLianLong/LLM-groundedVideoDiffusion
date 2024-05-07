import torch
import torch.nn.functional as F
import math
from collections.abc import Iterable
import warnings

import utils
from .attn import GaussianSmoothing

import re

import inflect

p = inflect.engine()


# A list mapping: prompt index to str (prompt in a list of token str)
def get_token_map(tokenizer, prompt, verbose=False, padding="do_not_pad"):
    fg_prompt_tokens = tokenizer(
        [prompt], padding=padding, max_length=77, return_tensors="np"
    )
    input_ids = fg_prompt_tokens["input_ids"][0]

    # index_to_last_with = np.max(np.where(input_ids == 593))
    # index_to_last_eot = np.max(np.where(input_ids == 49407))

    token_map = []
    for ind, item in enumerate(input_ids.tolist()):
        token = tokenizer._convert_id_to_token(item)
        if verbose:
            print(f"{ind}, {token} ({item})")

        token_map.append(token)

        # If we don't pad, we don't need to break.
        # if item == tokenizer.eos_token_id:
        #     break

    return token_map


def refine_phrase(prompt, phrase, verbose=False):
    refined_phrase = phrase
    found = True
    if not re.search(r"\b" + refined_phrase + r"\b", prompt):
        # This only checks substring instead of token matching so there might be corner cases.
        refined_phrase = refined_phrase.strip("0123456789 ")
        if not re.search(r"\b" + refined_phrase + r"\b", prompt):
            last_word = refined_phrase.split(" ")[-1]
            if verbose:
                print(
                    f'**Phrase "{refined_phrase}" is not part of the prompt "{prompt}", using the last word "{last_word}" only**'
                )
            refined_phrase = last_word
            if not re.search(r"\b" + refined_phrase + r"\b", prompt):
                refined_phrase = p.plural(refined_phrase)
                if verbose:
                    print(
                        f'**Phrase is still not part of the prompt "{prompt}", using plural "{refined_phrase}" instead**'
                    )
                if not re.search(r"\b" + refined_phrase + r"\b", prompt):
                    print(f'**Phrase is still not part of the prompt "{prompt}"**')
                    found = False

    return found, refined_phrase


def get_phrase_indices(
    tokenizer,
    prompt,
    phrases,
    verbose=False,
    words=None,
    include_eos=False,
    token_map=None,
    return_word_token_indices=False,
):
    if token_map is None:
        # We allow using a pre-computed token map.
        token_map = get_token_map(
            tokenizer, prompt=prompt, verbose=verbose, padding="do_not_pad"
        )
    token_map_str = " ".join(token_map)

    object_positions = []
    word_token_indices = []
    for obj_ind, obj in enumerate(phrases):
        found, refined_phrase = refine_phrase(prompt, obj)
        # This should not happen since if it's not in the prompt we will suffix the prompt with the object name
        assert found, "phrase {obj} not found in the prompt {prompt}"
        obj = refined_phrase

        phrase_token_map = get_token_map(
            tokenizer, prompt=obj, verbose=verbose, padding="do_not_pad"
        )

        # Remove <bos> and <eos> in substr
        phrase_token_map = phrase_token_map[1:-1]
        phrase_token_map_len = len(phrase_token_map)
        phrase_token_map_str = " ".join(phrase_token_map)

        if verbose:
            print(
                "Full str:",
                token_map_str,
                "Substr:",
                phrase_token_map_str,
                "Phrase:",
                phrases,
            )

        # Count the number of token before substr
        # The substring comes with a trailing space that needs to be removed by minus one in the index.
        obj_first_index = len(
            token_map_str[: token_map_str.index(phrase_token_map_str) - 1].split(" ")
        )

        obj_position = list(
            range(obj_first_index, obj_first_index + phrase_token_map_len)
        )
        if include_eos:
            obj_position.append(token_map.index(tokenizer.eos_token))
        object_positions.append(obj_position)

        if return_word_token_indices:
            # Picking the last token in the specification
            if words is None:
                so_token_index = object_positions[0][-1]
                # Picking the noun or perform pooling on attention with the tokens may be better
                print(
                    f'Picking the last token "{token_map[so_token_index]}" ({so_token_index}) as attention token for extracting attention for SAM, which might not be the right one'
                )
            else:
                word = words[obj_ind]
                word_token_map = get_token_map(
                    tokenizer, prompt=word, verbose=verbose, padding="do_not_pad"
                )
                # Get the index of the last token of word (the occurrence in phrase) in the prompt. Note that we skip the <eos> token through indexing with -2.
                so_token_index = obj_first_index + phrase_token_map.index(
                    word_token_map[-2]
                )

            if verbose:
                print("so_token_index:", so_token_index)

            word_token_indices.append(so_token_index)

    if return_word_token_indices:
        return object_positions, word_token_indices

    return object_positions


def center_of_mass(x, h_range, w_range):
    com_h = (x.sum(dim=2) * h_range).sum(dim=-1) / (x.sum(dim=(1, 2)))
    com_w = (x.sum(dim=1) * w_range).sum(dim=-1) / (x.sum(dim=(1, 2)))
    return com_h, com_w


def add_ca_loss_per_attn_map_to_loss(
    loss,
    attn_map,
    object_number,
    bboxes,
    object_positions,
    use_ratio_based_loss=False,
    use_max_based_loss=True,
    use_ce_based_loss=False,
    fg_top_p=0.2,
    bg_top_p=0.2,
    fg_weight=1.0,
    bg_weight=1.0,
    eps=1.0e-2,
    index=None,
    attn_key=None,
    exclude_bg_heads=False,
    smooth_attn=False,
    kernel_size=3,
    sigma=0.5,
    upsample_scale=1,
    upsample_mode="bilinear",
    base_attn_dim=(40, 72),
    attn_renorm=False,
    num_tokens=None,
    renorm_scale=2.0,
    attn_sync_weight=0.0,
    boxdiff_loss_scale=0.0,
    boxdiff_normed=True,
    boxdiff_L=1,
    com_loss_scale=0.0,
    verbose=False,
):
    """
    fg_top_p, bg_top_p, fg_weight, and bg_weight are only used with max-based loss

    NOTE: default loss is max based now.

    `index` (timestep) is for debugging.
    """

    assert not exclude_bg_heads, "exclude_bg_heads is deprecated"

    # Uncomment to debug:
    # print(fg_top_p, bg_top_p, fg_weight, bg_weight)
    # Example attn shape: [24, 20, 180, 77]
    # b: the number of heads
    n_f, b, i, j = attn_map.shape

    if smooth_attn:
        # print(f"**Attn is smoothed: kernel size {kernel_size}, sigma {sigma}")
        smoothing = GaussianSmoothing(
            channels=b, kernel_size=kernel_size, sigma=sigma, dim=2
        ).cuda()
        # print(f"Attn map (before): {attn_map.shape}")
        attn_map = F.pad(attn_map, (1, 1, 1, 1), mode="reflect")
        attn_map = smoothing(attn_map)
        # print(f"Attn map (after): {attn_map.shape}")
        assert attn_map.shape == (n_f, b, i, j), f"{attn_map.shape} != {(n_f, b, i, j)}"

    if attn_renorm:
        attn_map = attn_map[..., 1 : num_tokens - 1] * renorm_scale
        attn_map = F.softmax(attn_map, dim=-1)

    H_attn, W_attn = utils.get_hw_from_attn_dim(attn_dim=i, base_attn_dim=base_attn_dim)

    H, W = int(H_attn * upsample_scale), int(W_attn * upsample_scale)

    h_range = torch.arange(H, device=attn_map.device, dtype=attn_map.dtype)[None]
    w_range = torch.arange(W, device=attn_map.device, dtype=attn_map.dtype)[None]

    for obj_idx in range(object_number):
        obj_loss = 0
        obj_boxes = bboxes[obj_idx]

        assert (
            n_f == len(obj_boxes)
        ), f"Number of frames {n_f} mismatches with number of frames in box condition {len(obj_boxes)}"
        for frame_ind, frame_boxes in enumerate(obj_boxes):
            mask = torch.zeros(size=(H, W), device="cuda")
            if boxdiff_loss_scale > 0:
                corner_mask_x = torch.zeros(size=(1, W), device="cuda")
                corner_mask_y = torch.zeros(size=(1, H), device="cuda")
            # t1 is the next frame
            mask_t1 = torch.zeros(size=(H, W), device="cuda")
            frame_ind_t1 = min(frame_ind + 1, len(obj_boxes) - 1)
            frame_boxes_t1 = obj_boxes[frame_ind_t1]

            # We support three level (one box per frame per phrase) and four level (multiple boxes per frame per phrase)
            assert not isinstance(
                frame_boxes[0], Iterable
            ), "We currently process different boxes with the same name separately."
            frame_boxes = [frame_boxes]

            assert not isinstance(
                frame_boxes_t1[0], Iterable
            ), "We currently process different boxes with the same name separately."
            frame_boxes_t1 = [frame_boxes_t1]

            for obj_box in frame_boxes:
                # x_min, y_min, x_max, y_max = int(obj_box[0] * W), int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
                x_min, y_min, x_max, y_max = utils.scale_proportion(obj_box, H=H, W=W)
                mask[y_min:y_max, x_min:x_max] = 1

                if boxdiff_loss_scale > 0:
                    corner_mask_x[
                        :, max(x_min - boxdiff_L, 0) : min(x_min + boxdiff_L + 1, W)
                    ] = 1.0
                    corner_mask_x[
                        :, max(x_max - boxdiff_L, 0) : min(x_max + boxdiff_L + 1, W)
                    ] = 1.0
                    corner_mask_y[
                        :, max(y_min - boxdiff_L, 0) : min(y_min + boxdiff_L + 1, H)
                    ] = 1.0
                    corner_mask_y[
                        :, max(y_max - boxdiff_L, 0) : min(y_max + boxdiff_L + 1, H)
                    ] = 1.0

            for obj_box in frame_boxes_t1:
                # x_min, y_min, x_max, y_max = int(obj_box[0] * W), int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
                x_min, y_min, x_max, y_max = utils.scale_proportion(obj_box, H=H, W=W)
                mask_t1[y_min:y_max, x_min:x_max] = 1

            for obj_position in object_positions[obj_idx]:
                # Could potentially optimize to compute this for loop in batch.
                # Could crop the ref cross attention before saving to save memory.

                # ca_map_obj shape: (b, H * W)
                if attn_renorm:
                    # Since we removed SOT, we subtract 1 from obj_position.
                    ca_map_obj = attn_map[frame_ind, :, :, obj_position - 1]
                    ca_map_obj_t1 = attn_map[frame_ind_t1, :, :, obj_position - 1]
                else:
                    ca_map_obj = attn_map[frame_ind, :, :, obj_position]
                    ca_map_obj_t1 = attn_map[frame_ind_t1, :, :, obj_position]

                # Note that the shape of ca_map_obj are different whether upsample_scale is 0 or not, but since we have reshape/view afterwards this does not cause a problem.
                if upsample_scale != 1:
                    # import ipdb; ipdb.set_trace()
                    # Interpolate each obj_position instead of the whole attention map to save memory
                    ca_map_obj = ca_map_obj.view(b, 1, H_attn, W_attn)

                    # ca_map_obj after interpolation: b, 1, H, W
                    ca_map_obj = F.interpolate(
                        ca_map_obj, size=(H, W), mode=upsample_mode
                    )

                    ca_map_obj_t1 = ca_map_obj_t1.view(b, 1, H_attn, W_attn)
                    ca_map_obj_t1 = F.interpolate(
                        ca_map_obj_t1, size=(H, W), mode=upsample_mode
                    )

                if use_ratio_based_loss:
                    ca_map_obj = ca_map_obj.reshape(b, H, W)
                    warnings.warn(
                        "Using ratio-based loss, which is deprecated. Max-based loss is recommended. The scale may be different."
                    )
                    # Ratio-based loss function

                    # Enforces the attention to be within the mask only. Does not enforce within-mask distribution.
                    activation_value = (ca_map_obj * mask).reshape(b, -1).sum(
                        dim=-1
                    ) / (ca_map_obj.reshape(b, -1).sum(dim=-1) + eps)
                    obj_loss += torch.mean((1 - activation_value) ** 2)
                else:
                    # This also accepts the case with mask set to 0 in all places (empty boxes). In this case, only background loss will be activated (discourages the object from appearing).
                    # However, this requires the boxes to not be filtered out in `filter_boxes`.
                    # The value is at least 1 in the size of the original attention mask.
                    k_fg = (
                        (mask.sum() * fg_top_p)
                        .long()
                        .clamp_(min=int(upsample_scale * upsample_scale))
                    )
                    k_bg = (
                        ((1 - mask).sum() * bg_top_p)
                        .long()
                        .clamp_(min=int(upsample_scale * upsample_scale))
                    )

                    mask_1d = mask.view(1, -1)

                    if use_max_based_loss:
                        # Max-based loss function

                        # Take the topk over spatial dimension, and then take the sum over heads dim
                        # The mean is over k_fg and k_bg dimension, so we don't need to sum and divide on our own.
                        fg_loss = (
                            1 - (ca_map_obj * mask_1d).topk(k=k_fg).values.mean(dim=1)
                        ).sum(dim=0)
                        obj_loss += fg_loss * fg_weight
                        bg_loss = (
                            (ca_map_obj * (1 - mask_1d)).topk(k=k_bg).values.mean(dim=1)
                        ).sum(dim=0)
                        obj_loss += bg_loss * bg_weight

                        # print(f"{attn_key} at index {index}: fg_loss: {fg_loss}, bg_loss: {bg_loss}, weighted sum: {fg_loss * fg_weight + bg_loss * bg_weight}")

                        # Debugging:
                        # print("ca_map_obj max", (ca_map_obj * (1 - mask_1d)).max())

                        # Uncomment to debug at timestep index 15
                        # if index == 15:
                        #     import ipdb; ipdb.set_trace()
                    elif use_ce_based_loss:
                        # Use CE loss (NLL)
                        # Take the topk over spatial dimension, and then take the sum over heads dim
                        # The mean is over k_fg and k_bg dimension, so we don't need to sum and divide on our own.
                        # clamp is for numerical stability
                        ca_map_obj = torch.clamp(ca_map_obj, min=eps, max=1 - eps)

                        # This clamp is needed to prevent the case the masks are all 0.
                        fg_loss = (
                            -torch.log(
                                torch.clamp(
                                    (mask_1d * ca_map_obj).topk(k=k_fg).values, min=eps
                                )
                            )
                            .mean(dim=1)
                            .sum(dim=0)
                        )
                        obj_loss += fg_loss * fg_weight
                        bg_loss = -torch.log(
                            1
                            - ((1 - mask_1d) * ca_map_obj)
                            .topk(k=k_bg)
                            .values.mean(dim=1)
                        ).sum(dim=0)
                        obj_loss += bg_loss * bg_weight

                        if torch.isinf(fg_loss) or torch.isinf(bg_loss):
                            print(
                                f"Encountered inf loss. fg_loss: {fg_loss}, bg_loss: {bg_loss}"
                            )

                        # print(f"Using NLL.")
                        # print(f"fg: {-torch.log((mask_1d * ca_map_obj).topk(k=k_fg).values).mean(dim=1).sum(dim=0) * fg_weight}, bg: {-torch.log(1 - ((1 - mask_1d) * ca_map_obj).topk(k=k_bg).values.mean(dim=1)).sum(dim=0) * bg_weight}")
                        # print(f"log: {(mask_1d * (torch.log(ca_map_obj)+100)).topk(k=k_fg).values-100}, without log: {torch.log((mask_1d * ca_map_obj).topk(k=k_fg).values)}")

                    else:
                        raise ValueError("Unknown loss: no loss selected")

                if attn_sync_weight != 0.0:
                    assert (
                        not attn_renorm
                    ), "attn_sync with attn_renorm not implemented together"
                    if frame_ind != len(obj_boxes) - 1:
                        # frame 1 (ca_map_obj) should not be the last frame
                        # b, i: b is the number of attention heads, i is the spatial dimension
                        ca_map_obj_frame2 = attn_map[frame_ind + 1, :, :, obj_position]

                        if len(frame_boxes) > 1:
                            print(
                                "***Warning: more than one box per object per frame. Will only sync the weights with the last box per object.***"
                            )

                        # Turn into 2D and crop

                        ca_map_obj_frame1_box = ca_map_obj.view(b, H_attn, W_attn)[
                            :, y_min:y_max, x_min:x_max
                        ]
                        ca_map_obj_frame2_box = ca_map_obj_frame2.view(
                            b, H_attn, W_attn
                        )[:, y_min:y_max, x_min:x_max]
                        # import ipdb; ipdb.set_trace()
                        attn_sync_loss = (
                            (ca_map_obj_frame1_box - ca_map_obj_frame2_box) ** 2
                        ).mean(dim=(1, 2)).sum(dim=0) * attn_sync_weight

                        # print(f"{attn_key} at index {index}: attn_sync_loss: {attn_sync_loss}")

                        obj_loss += attn_sync_loss

                # corner constraint loss in BoxDiff
                if boxdiff_loss_scale > 0:
                    # ca_map_obj_max_x: (b, W)
                    # ca_map_obj_max_y: (b, H)
                    ca_map_obj = ca_map_obj.view(b, H, W)
                    ca_map_obj_max_x = ca_map_obj.max(dim=1).values.to(torch.float32)
                    ca_map_obj_max_y = ca_map_obj.max(dim=2).values.to(torch.float32)
                    # mask: (H, W)
                    # mask_max_x: (1, W)
                    # mask_max_y: (1, H)
                    mask_max_x = mask[None].max(dim=1).values
                    mask_max_y = mask[None].max(dim=2).values

                    # corner_mask_x: (1, W)
                    # corner_mask_y: (1, H)
                    if boxdiff_normed:
                        cc_loss = (
                            (ca_map_obj_max_x - mask_max_x).abs() * corner_mask_x
                        ).mean() + (
                            (ca_map_obj_max_y - mask_max_y).abs() * corner_mask_y
                        ).mean()
                    else:
                        cc_loss = (
                            (ca_map_obj_max_x - mask_max_x).abs() * corner_mask_x
                        ).sum() + (
                            (ca_map_obj_max_y - mask_max_y).abs() * corner_mask_y
                        ).sum()

                    # original implementation (no corner mask, used norm which sums up all the points)
                    # cc_loss = (ca_map_obj_max_x - mask_max_x).norm(p=1) + (ca_map_obj_max_y - mask_max_y).norm(p=1)
                    # This is equivalent to:
                    # cc_loss = ((ca_map_obj_max_x - mask_max_x).abs()).sum() + ((ca_map_obj_max_y - mask_max_y).abs()).sum()

                    obj_loss += cc_loss * boxdiff_loss_scale

                # center of mass (com) loss
                if com_loss_scale > 0:
                    # position control
                    if mask.sum() > 0:
                        ca_map_obj = ca_map_obj.view(b, H, W)
                        com_ca_h, com_ca_w = center_of_mass(
                            ca_map_obj.to(torch.float32),
                            h_range=h_range,
                            w_range=w_range,
                        )
                        com_mask_h, com_mask_w = center_of_mass(
                            mask[None], h_range=h_range, w_range=w_range
                        )
                        # com_loss = F.mse_loss(com_ca_h, com_mask_h) + F.mse_loss(com_ca_w, com_mask_w)
                        com_loss = (
                            (com_ca_h[..., None, None] - com_mask_h) ** 2
                        ).mean() + (
                            (com_ca_w[..., None, None] - com_mask_w) ** 2
                        ).mean()
                        obj_loss += com_loss * com_loss_scale

                        # velocity control
                        if mask_t1.sum() > 0:
                            ca_map_obj = ca_map_obj.view(b, H, W)
                            ca_map_obj_t1 = ca_map_obj_t1.view(b, H, W)
                            com_ca_h, com_ca_w = center_of_mass(
                                ca_map_obj.to(torch.float32),
                                h_range=h_range,
                                w_range=w_range,
                            )
                            com_ca_h_t1, com_ca_w_t1 = center_of_mass(
                                ca_map_obj_t1.to(torch.float32),
                                h_range=h_range,
                                w_range=w_range,
                            )
                            com_mask_h, com_mask_w = center_of_mass(
                                mask[None], h_range=h_range, w_range=w_range
                            )
                            com_mask_h_t1, com_mask_w_t1 = center_of_mass(
                                mask_t1[None], h_range=h_range, w_range=w_range
                            )
                            # com_loss = F.mse_loss(com_ca_h_t1 - com_ca_h, com_mask_h_t1 - com_mask_h) + F.mse_loss(com_ca_w_t1 - com_ca_w, com_mask_w_t1 - com_mask_w)
                            com_loss = (
                                (
                                    (com_ca_h_t1 - com_ca_h)[..., None, None]
                                    - (com_mask_h_t1 - com_mask_h)
                                )
                                ** 2
                            ).mean() + (
                                (
                                    (com_ca_w_t1 - com_ca_w)[..., None, None]
                                    - (com_mask_w_t1 - com_mask_w)
                                )
                                ** 2
                            ).mean()
                            obj_loss += com_loss * com_loss_scale

        loss += obj_loss / len(object_positions[obj_idx])

    return loss


def compute_ca_lossv3(
    saved_attn,
    bboxes,
    object_positions,
    guidance_attn_keys,
    index=None,
    verbose=False,
    **kwargs,
):
    """
    The `saved_attn` is supposed to be passed to `save_attn_to_dict` in `cross_attention_kwargs` prior to computing ths loss.
    `AttnProcessor` will put attention maps into the `save_attn_to_dict`.

    `index` is the timestep.
    """
    loss = torch.tensor(0).float().cuda()
    object_number = len(bboxes)
    if object_number == 0:
        return loss

    for attn_key in guidance_attn_keys:
        # We only have 1 cross attention for mid.
        attn_map_integrated = saved_attn[attn_key]
        if not attn_map_integrated.is_cuda:
            attn_map_integrated = attn_map_integrated.cuda()
        # Example dimension: [20, 64, 77]
        attn_map = attn_map_integrated.squeeze(dim=0)

        loss = add_ca_loss_per_attn_map_to_loss(
            loss,
            attn_map,
            object_number,
            bboxes,
            object_positions,
            attn_key=attn_key,
            index=index,
            verbose=verbose,
            **kwargs,
        )

    num_attn = len(guidance_attn_keys)

    if num_attn > 0:
        loss = loss / (object_number * num_attn)

    return loss
