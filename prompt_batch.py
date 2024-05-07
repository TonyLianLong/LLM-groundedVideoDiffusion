import os
from prompt import get_prompts, template_versions
from utils import parse, utils
from utils.parse import show_video_boxes, size
from utils.llm import get_llm_kwargs, get_full_prompt, model_names, get_parsed_layout
from utils import cache
import argparse
import time

# This only applies to visualization in this file.
scale_boxes = False

if scale_boxes:
    print("Scaling the bounding box to fit the scene")
else:
    print("Not scaling the bounding box to fit the scene")

H, W = size


def visualize_layout(parsed_layout):
    condition = parse.parsed_layout_to_condition(
        parsed_layout, tokenizer=None, height=H, width=W, verbose=True
    )

    show_video_boxes(condition, ind=ind, save=True)

    print(f"Visualize masks at {parse.img_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-type", type=str, default="demo")
    parser.add_argument("--model", choices=model_names, required=True)
    parser.add_argument("--template_version", choices=template_versions, required=True)
    parser.add_argument(
        "--auto-query", action="store_true", help="Auto query using the API"
    )
    parser.add_argument(
        "--always-save",
        action="store_true",
        help="Always save the layout without confirming",
    )
    parser.add_argument("--no-visualize", action="store_true", help="No visualizations")
    parser.add_argument(
        "--visualize-cache-hit", action="store_true", help="Save boxes for cache hit"
    )
    parser.add_argument(
        "--unnormalize-boxes-before-save",
        action="store_true",
        help="Unnormalize the boxes before saving. This should be enabled if the prompt asks the LLM to return normalized boxes.",
    )
    args = parser.parse_args()

    visualize_cache_hit = args.visualize_cache_hit

    template_version = args.template_version

    model, llm_kwargs = get_llm_kwargs(
        model=args.model, template_version=template_version
    )
    template = llm_kwargs.template
    # Need to parse json format for json templates
    json_template = "json" in template_version

    # This is for visualizing bounding boxes
    parse.img_dir = (
        f"img_generations/imgs_{args.prompt_type}_template{template_version}"
    )
    if not args.no_visualize:
        os.makedirs(parse.img_dir, exist_ok=True)

    cache.cache_path = f'cache/cache_{args.prompt_type.replace("lmd_", "")}{"_" + template_version if args.template_version != "v5" else ""}_{model}.json'

    os.makedirs(os.path.dirname(cache.cache_path), exist_ok=True)
    cache.cache_format = "json"

    cache.init_cache()

    prompts_query = get_prompts(args.prompt_type)

    max_attempts = 1

    for ind, prompt in enumerate(prompts_query):
        if isinstance(prompt, list):
            # prompt, seed
            prompt = prompt[0]
        prompt = prompt.strip().rstrip(".")

        resp = cache.get_cache(prompt)
        if resp is None:
            print(f"Cache miss: {prompt}")

            if not args.auto_query:
                print("#########")
                prompt_full = get_full_prompt(template=template, prompt=prompt)
                print(prompt_full)
                print("#########")
                resp = None

            attempts = 0
            while True:
                attempts += 1
                try:
                    # The resp from `get_parsed_layout` has already been structured
                    if args.auto_query:
                        parsed_layout, resp = get_parsed_layout(
                            prompt,
                            llm_kwargs=llm_kwargs,
                            json_template=json_template,
                            verbose=False,
                        )
                        print("Response:", resp)
                    else:
                        resp = utils.multiline_input(
                            prompt="Please enter LLM response (use an empty line to end): "
                        )
                        parsed_layout, resp = get_parsed_layout(
                            prompt,
                            llm_kwargs=llm_kwargs,
                            override_response=resp,
                            max_partial_response_retries=1,
                            json_template=json_template,
                            verbose=False,
                        )

                except (ValueError, SyntaxError, TypeError) as e:
                    if attempts > max_attempts:
                        print("Retrying too many times, skipping")
                        break
                    print(
                        f"Encountered invalid data with prompt {prompt} and response {resp}: {e}, retrying"
                    )
                    time.sleep(1)
                    continue

                if not args.no_visualize:
                    visualize_layout(parsed_layout)
                if not args.always_save:
                    save = input("Save (y/n)? ").strip()
                else:
                    save = "y"
                if save == "y" or save == "Y":
                    cache.add_cache(prompt, resp)
                else:
                    print("Not saved. Will generate the same prompt again.")
                    continue
                break
        else:
            print(f"Cache hit: {prompt}")

            parsed_layout, resp = get_parsed_layout(
                prompt,
                llm_kwargs=llm_kwargs,
                override_response=resp,
                max_partial_response_retries=1,
                json_template=json_template,
                verbose=False,
            )

            if visualize_cache_hit:
                visualize_layout(parsed_layout)
