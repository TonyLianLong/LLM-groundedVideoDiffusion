from utils import parse, vis, cache
from utils.llm import get_full_model_name, model_names, get_parsed_layout
from utils.parse import show_video_boxes, size
from tqdm import tqdm
import os
from prompt import get_prompts, template_versions
import matplotlib.pyplot as plt
import traceback
import bdb
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--save-suffix", default=None, type=str)
parser.add_argument(
    "--model",
    choices=model_names,
    required=True,
    help="LLM model to load the cache from",
)
parser.add_argument(
    "--repeats", default=1, type=int, help="Number of samples for each prompt"
)
parser.add_argument(
    "--regenerate",
    default=1,
    type=int,
    help="Number of regenerations. Different from repeats, regeneration happens after everything is generated",
)
parser.add_argument(
    "--force_run_ind",
    default=None,
    type=int,
    help="If this is enabled, we use this run_ind and skips generated images. If this is not enabled, we create a new run after existing runs.",
)
parser.add_argument(
    "--skip_first_prompts",
    default=0,
    type=int,
    help="Skip the first prompts in generation (useful for parallel generation)",
)
parser.add_argument(
    "--seed_offset",
    default=0,
    type=int,
    help="Offset to the seed (seed starts from this number)",
)
parser.add_argument(
    "--num_prompts",
    default=None,
    type=int,
    help="The number of prompts to generate (useful for parallel generation)",
)
parser.add_argument(
    "--run-model",
    default="lvd",
    choices=[
        "lvd",
        "lvd_zeroscope",
        "lvd_modelscope256",
        "lvd-gligen_modelscope256",
        "lvd-gligen_zeroscope",
        "lvd-plus_modelscope256",
        "lvd_modelscope512",
        "modelscope",
        "modelscope_256",
        "zeroscope",
        "zeroscope_xl",
    ],
    help="The model to use (modelscope has the option to generate with resolution 256x256)",
)
parser.add_argument("--visualize", action="store_true")
parser.add_argument("--no-continue-on-error", action="store_true")
parser.add_argument("--prompt-type", type=str, default="demo")
parser.add_argument("--template_version", choices=template_versions, required=True)
parser.add_argument("--dry-run", action="store_true", help="skip the generation")

float_args = [
    "fg_top_p",
    "bg_top_p",
    "fg_weight",
    "bg_weight",
    "loss_threshold",
    "loss_scale",
    "boxdiff_loss_scale",
    "com_loss_scale",
    "gligen_scheduled_sampling_beta",
]
for float_arg in float_args:
    parser.add_argument("--" + float_arg, default=None, type=float)

# `use_ratio_based_loss` should be 0 or 1 (as it is a bool)
int_args = [
    "num_inference_steps",
    "max_iter",
    "max_index_step",
    "num_frames",
    "use_ratio_based_loss",
    "boxdiff_normed",
]
for int_arg in int_args:
    parser.add_argument("--" + int_arg, default=None, type=int)

str_args = []
for str_arg in str_args:
    parser.add_argument("--" + str_arg, default=None, type=str)

args = parser.parse_args()


if not args.dry_run:
    run_model = args.run_model
    baseline = run_model in [
        "modelscope",
        "zeroscope",
        "modelscope_256",
        "zeroscope_xl",
    ]

    if "_" in run_model:
        option = run_model.split("_")[1]
    else:
        option = ""

    if run_model.startswith("lvd-plus"):
        import generation.lvd_plus as generation

        base_model = option if option else "modelscope"
        H, W = generation.init(base_model=base_model)
    elif run_model.startswith("lvd-gligen"):
        import generation.lvd_gligen as generation

        base_model = option if option else "modelscope"
        H, W = generation.init(base_model=base_model)
    elif (run_model == "lvd") or (run_model.startswith("lvd_")):
        import generation.lvd as generation

        # Use modelscope as the default model
        base_model = option if option else "modelscope"
        H, W = generation.init(base_model=base_model)
    elif run_model == "modelscope" or run_model == "modelscope_256":
        import generation.modelscope_dpm as generation

        H, W = generation.init(option=option)
    elif run_model == "zeroscope" or run_model == "zeroscope_xl":
        import generation.zeroscope_dpm as generation

        H, W = generation.init(option=option)
    else:
        raise ValueError(f"Unknown model: {run_model}")

    if "zeroscope" in run_model and (
        (args.num_frames is not None and args.num_frames < 24)
        or ((not baseline) and args.num_frames is None)
    ):
        # num_frames is 16 by default in non-baseline models (as it uses modelscope by default)
        raise ValueError(
            "Running zeroscope with fewer than 24 frames. This may lead to suboptimal results. Comment this out if you still want to run."
        )

    version = generation.version
    assert (
        version == args.run_model.split("_")[0]
    ), f"{version} != {args.run_model.split('_')[0]}"
    run = generation.run
else:
    version = "dry_run"
    run = None
    generation = argparse.Namespace()

# set visualizations to no-op in batch generation
for k in vis.__dict__.keys():
    if k.startswith("visualize"):
        vis.__dict__[k] = lambda *args, **kwargs: None


## Visualize
def visualize_layout(parsed_layout):
    H, W = size
    condition = parse.parsed_layout_to_condition(
        parsed_layout, tokenizer=None, height=H, width=W, verbose=True
    )

    show_video_boxes(condition, ind=ind, save=True)

    print(f"Visualize masks at {parse.img_dir}")


# close the figure when plt.show is called
plt.show = plt.close

prompt_type = args.prompt_type
template_version = args.template_version
json_template = "json" in template_version

# Use cache
model = get_full_model_name(model=args.model)

if not baseline:
    cache.cache_format = "json"
    cache.cache_path = f'cache/cache_{args.prompt_type.replace("lmd_", "")}_{template_version}_{model}.json'
    print(f"Loading LLM responses from cache {cache.cache_path}")
    cache.init_cache(allow_nonexist=False)

prompts = get_prompts(prompt_type)

save_suffix = ("_" + args.save_suffix) if args.save_suffix else ""
repeats = args.repeats
seed_offset = args.seed_offset

model_in_base_save_dir = "" if model == "gpt-4" else f"_{model}"
base_save_dir = f"img_generations/imgs_{prompt_type}_template{args.template_version}{model_in_base_save_dir}_{run_model}{save_suffix}"

run_kwargs = {}

argnames = float_args + int_args + str_args

for argname in argnames:
    argvalue = getattr(args, argname)
    if argvalue is not None:
        run_kwargs[argname] = argvalue

is_notebook = False

if args.force_run_ind is not None:
    run_ind = args.force_run_ind
    save_dir = f"{base_save_dir}/run{run_ind}"
else:
    run_ind = 0
    while True:
        save_dir = f"{base_save_dir}/run{run_ind}"
        if not os.path.exists(save_dir):
            break
        run_ind += 1

if hasattr(generation, "use_autocast") and generation.use_autocast:
    save_dir += "_amp"

print(f"Save dir: {save_dir}")

LARGE_CONSTANT = 123456789
LARGE_CONSTANT2 = 56789
LARGE_CONSTANT3 = 6789

ind = 0
if args.regenerate > 1:
    # Need to fix the ind
    assert args.skip_first_prompts == 0

for regenerate_ind in range(args.regenerate):
    print("regenerate_ind:", regenerate_ind)
    if not baseline:
        cache.reset_cache_access()
    for prompt_ind, prompt in enumerate(tqdm(prompts, desc=f"Run: {save_dir}")):
        if prompt_ind < args.skip_first_prompts:
            ind += 1
            continue
        if args.num_prompts is not None and prompt_ind >= (
            args.skip_first_prompts + args.num_prompts
        ):
            ind += 1
            continue

        # get prompt from prompts, if prompt is a list, then prompt includes both the prompt and kwargs
        if isinstance(prompt, list):
            prompt, kwargs = prompt
        else:
            kwargs = {}

        prompt = prompt.strip().rstrip(".")

        ind_override = kwargs.get("seed", None)

        # Load from cache
        if baseline:
            resp = None
        else:
            resp = cache.get_cache(prompt)

            if resp is None:
                print(f"Cache miss, skipping prompt: {prompt}")
                ind += 1
                continue

        print(f"***run: {run_ind}***")
        print(f"prompt: {prompt}, resp: {resp}")
        parse.img_dir = f"{save_dir}/{ind}"
        # Skip if image is already generared
        if not (
            os.path.exists(parse.img_dir)
            and len(
                [
                    img
                    for img in os.listdir(parse.img_dir)
                    if img.startswith("video") and img.endswith("joblib")
                ]
            )
            >= args.repeats
        ):
            os.makedirs(parse.img_dir, exist_ok=True)
            try:
                if baseline:
                    parsed_layout = {"Prompt": prompt}
                else:
                    parsed_layout, _ = get_parsed_layout(
                        prompt,
                        max_partial_response_retries=1,
                        override_response=resp,
                        json_template=json_template,
                    )

                print("parsed_layout:", parsed_layout)

                if args.dry_run:
                    # Skip generation
                    ind += 1
                    continue

                if args.visualize:
                    assert (
                        not baseline
                    ), "baseline methods do not have layouts from the LLM to visualize"
                    visualize_layout(parsed_layout)

                original_ind_base = (
                    ind_override + regenerate_ind * LARGE_CONSTANT2
                    if ind_override is not None
                    else ind
                )

                for repeat_ind in range(repeats):
                    ind_offset = repeat_ind * LARGE_CONSTANT3 + seed_offset
                    run(
                        parsed_layout,
                        seed=original_ind_base + ind_offset,
                        repeat_ind=repeat_ind,
                        **run_kwargs,
                    )

            except (KeyboardInterrupt, bdb.BdbQuit) as e:
                print(e)
                exit()
            except RuntimeError:
                print(
                    "***RuntimeError: might run out of memory, skipping the current one***"
                )
                print(traceback.format_exc())
                time.sleep(10)
            except Exception as e:
                print(f"***Error: {e}***")
                print(traceback.format_exc())
                if args.no_continue_on_error:
                    raise e
        else:
            print(f"Image exists at {parse.img_dir}, skipping")
        ind += 1

    if not baseline and cache.values_accessed() != len(prompts):
        print(
            f"**Cache is hit {cache.values_accessed()} time(s) but we have {len(prompts)} prompts. There may be cache misses or inconsistencies between the prompts and the cache such as extra items in the cache.**"
        )
