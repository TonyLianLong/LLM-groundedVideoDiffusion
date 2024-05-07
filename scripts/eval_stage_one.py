# This script allows evaluating stage one and saving the generated prompts to cache

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import argparse
from prompt import get_prompts, get_num_parsed_layout_frames, template_versions
from utils.llm import get_llm_kwargs, get_parsed_layout_with_cache, model_names
from utils.eval import evaluate_with_layout
from utils import parse, cache
import numpy as np
from tqdm import tqdm

eval_success_counts = {}
eval_all_counts = {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-type", type=str, default="lvd")
    parser.add_argument("--model", choices=model_names, required=True)
    parser.add_argument("--template_version", choices=template_versions, required=True)
    parser.add_argument("--skip_first_prompts", default=0, type=int)
    parser.add_argument("--num_prompts", default=None, type=int)
    parser.add_argument("--show-cache-access", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    np.set_printoptions(precision=2)

    template_version = args.template_version

    json_template = "json" in template_version

    model, llm_kwargs = get_llm_kwargs(
        model=args.model, template_version=template_version
    )

    cache.cache_format = "json"
    cache.cache_path = f'cache/cache_{args.prompt_type.replace("lmd_", "")}_{template_version}_{model}.json'
    cache.init_cache()

    prompt_predicates = get_prompts(args.prompt_type, return_predicates=True)
    print(f"Number of prompts (predicates): {len(prompt_predicates)}")

    height, width = parse.size_h, parse.size_w

    for ind, (prompt, predicate) in enumerate(tqdm(prompt_predicates)):
        if isinstance(prompt, list):
            # prompt and kwargs
            prompt = prompt[0]
        prompt = prompt.strip().rstrip(".")
        if ind < args.skip_first_prompts:
            continue
        if args.num_prompts is not None and ind >= (
            args.skip_first_prompts + args.num_prompts
        ):
            continue

        parsed_layout = get_parsed_layout_with_cache(
            prompt, llm_kwargs, json_template=json_template, verbose=args.verbose
        )
        num_parsed_layout_frames = get_num_parsed_layout_frames(template_version)
        eval_type, eval_success = evaluate_with_layout(
            parsed_layout,
            predicate,
            num_parsed_layout_frames,
            height=height,
            width=width,
            verbose=args.verbose,
        )

        print(f"Eval success (eval_type):", eval_success)

        if eval_type not in eval_all_counts:
            eval_success_counts[eval_type] = 0
            eval_all_counts[eval_type] = 0
        eval_success_counts[eval_type] += int(eval_success)
        eval_all_counts[eval_type] += 1

    eval_success_conut, eval_all_count = 0, 0
    for k, v in eval_all_counts.items():
        print(
            f"Eval type: {k}, success: {eval_success_counts[k]}/{eval_all_counts[k]}, rate: {eval_success_counts[k]/eval_all_counts[k]:.2f}"
        )
        eval_success_conut += eval_success_counts[k]
        eval_all_count += eval_all_counts[k]

    print(
        f"Overall: success: {eval_success_conut}/{eval_all_count}, rate: {eval_success_conut/eval_all_count:.2f}"
    )

    if args.show_cache_access:
        # Print what are accessed in the cache (may have multiple values in each key)
        # Not including the newly added items
        print(json.dumps(cache.cache_queries))
        print("Number of accessed keys:", len(cache.cache_queries))
