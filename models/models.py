import torch

# For compatibility
from utils import torch_device


def encode_prompts(
    tokenizer,
    text_encoder,
    prompts,
    negative_prompt="",
    return_full_only=False,
    one_uncond_input_only=False,
):
    if negative_prompt == "":
        print("Note that negative_prompt is an empty string")

    text_input = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    max_length = text_input.input_ids.shape[-1]
    if one_uncond_input_only:
        num_uncond_input = 1
    else:
        num_uncond_input = len(prompts)
    uncond_input = tokenizer(
        [negative_prompt] * num_uncond_input,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
        cond_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

    if one_uncond_input_only:
        return uncond_embeddings, cond_embeddings

    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

    if return_full_only:
        return text_embeddings
    return text_embeddings, uncond_embeddings, cond_embeddings


def process_input_embeddings(input_embeddings):
    assert isinstance(input_embeddings, (tuple, list))
    if len(input_embeddings) == 3:
        # input_embeddings: text_embeddings, uncond_embeddings, cond_embeddings
        # Assume `uncond_embeddings` is full (has batch size the same as cond_embeddings)
        _, uncond_embeddings, cond_embeddings = input_embeddings
        assert (
            uncond_embeddings.shape[0] == cond_embeddings.shape[0]
        ), f"{uncond_embeddings.shape[0]} != {cond_embeddings.shape[0]}"
        return input_embeddings
    elif len(input_embeddings) == 2:
        # input_embeddings: uncond_embeddings, cond_embeddings
        # uncond_embeddings may have only one item
        uncond_embeddings, cond_embeddings = input_embeddings
        if uncond_embeddings.shape[0] == 1:
            uncond_embeddings = uncond_embeddings.expand(cond_embeddings.shape)
        # We follow the convention: negative (unconditional) prompt comes first
        text_embeddings = torch.cat((uncond_embeddings, cond_embeddings), dim=0)
        return text_embeddings, uncond_embeddings, cond_embeddings
    else:
        raise ValueError(f"input_embeddings length: {len(input_embeddings)}")


def attn_list_to_tensor(cross_attention_probs):
    # timestep, CrossAttnBlock, Transformer2DModel, 1xBasicTransformerBlock

    num_cross_attn_block = len(cross_attention_probs[0])
    cross_attention_probs_all = []

    for i in range(num_cross_attn_block):
        # cross_attention_probs_timestep[i]: Transformer2DModel
        # 1xBasicTransformerBlock is skipped
        cross_attention_probs_current = []
        for cross_attention_probs_timestep in cross_attention_probs:
            cross_attention_probs_current.append(
                torch.stack([item for item in cross_attention_probs_timestep[i]], dim=0)
            )

        cross_attention_probs_current = torch.stack(
            cross_attention_probs_current, dim=0
        )
        cross_attention_probs_all.append(cross_attention_probs_current)

    return cross_attention_probs_all
