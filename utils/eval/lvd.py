import numpy as np
from functools import partial
from .utils import (
    p,
    predicate_numeracy,
    predicate_attribution,
    predicate_visibility,
    predicate_1obj_dynamic_spatial,
    predicate_2obj_dynamic_spatial,
    predicate_sequentialv2,
)

prompt_prefix = "A realistic lively video of a scene"
prompt_top_down_prefix = "A realistic lively video of a top-down viewed scene"

evaluate_classes = [
    ("moving car", "car"),
    ("lively cat", "cat"),
    ("flying bird", "bird"),
    ("moving ball", "ball"),
    ("walking dog", "dog"),
]
evaluate_classes_no_attribute = [
    evaluate_class_no_attribute
    for evaluate_class, evaluate_class_no_attribute in evaluate_classes
]


def get_prompt_predicates_numeracy(min_num=1, max_num=5, repeat=2):
    modifier = ""

    prompt_predicates = []

    for number in range(min_num, max_num + 1):
        for object_name, object_name_no_attribute in evaluate_classes:
            if prompt_prefix:
                prompt = f"{prompt_prefix} with {p.number_to_words(number) if number < 21 else number}{modifier} {p.plural(object_name) if number > 1 else object_name}"
            else:
                prompt = f"{p.number_to_words(number) if number < 21 else number}{modifier} {p.plural(object_name) if number > 1 else object_name}"
            prompt = prompt.strip()

            # `query_names` needs to match with `texts` since `query_names` will be searched in the detection of `texts`
            query_names = (object_name_no_attribute,)
            predicate = partial(predicate_numeracy, query_names, number)
            predicate.type = "numeracy"
            predicate.texts = [f"a photo of {p.a(object_name_no_attribute)}"]
            # We don't have tracking, but mismatch does not matter for numeracy.
            predicate.one_box_per_class = False
            prompt_predicate = prompt, predicate

            prompt_predicates += [prompt_predicate] * repeat

    return prompt_predicates


def process_object_name(object_name):
    if isinstance(object_name, tuple):
        query_names = object_name
        object_name = object_name[0]
    else:
        query_names = (object_name,)

    return object_name, query_names


def get_prompt_predicates_attribution(num_prompts=100, repeat=1):
    prompt_predicates = []

    intended_count1, intended_count2 = 1, 1

    modifiers = [
        "red",
        "orange",
        "yellow",
        "green",
        "blue",
        "purple",
        "pink",
        "brown",
        "black",
        "white",
        "gray",
    ]

    for ind in range(num_prompts):
        np.random.seed(ind)
        modifier1, modifier2 = np.random.choice(modifiers, 2, replace=False)
        object_name1, object_name2 = np.random.choice(
            evaluate_classes_no_attribute, 2, replace=False
        )

        object_name1, query_names1 = process_object_name(object_name1)
        object_name2, query_names2 = process_object_name(object_name2)

        if prompt_prefix:
            prompt = f"{prompt_prefix} with {p.a(modifier1)} {object_name1} and {p.a(modifier2)} {object_name2}"
        else:
            prompt = (
                f"{p.a(modifier1)} {object_name1} and {p.a(modifier2)} {object_name2}"
            )
        prompt = prompt.strip()

        # `query_names` needs to match with `texts` since `query_names` will be searched in the detection of `texts`
        predicate = partial(
            predicate_attribution,
            query_names1,
            query_names2,
            modifier1,
            modifier2,
            intended_count1,
            intended_count2,
        )

        prompt_predicate = prompt, predicate

        predicate.type = "attribution"
        predicate.texts = [
            f"a photo of {p.a(modifier1)} {object_name1}",
            f"a photo of {p.a(modifier2)} {object_name2}",
        ]
        # Limit to one box per class.
        predicate.one_box_per_class = True

        prompt_predicates += [prompt_predicate] * repeat

    return prompt_predicates


def get_prompt_predicates_visibility(repeat=2):
    prompt_predicates = []

    for object_name, object_name_no_attribute in evaluate_classes:
        # `query_names` needs to match with `texts` since `query_names` will be searched in the detection of `texts`
        query_names = (object_name_no_attribute,)

        for i in range(2):
            # i == 0: appeared
            # i == 1: disappeared

            if i == 0:
                prompt = f"{prompt_prefix} in which {p.a(object_name)} appears only in the second half of the video"
                # Shouldn't use lambda here since query_names (and number) might change.
                predicate = partial(predicate_visibility, query_names, True)
                prompt_predicate = prompt, predicate
            else:
                prompt = f"{prompt_prefix} in which {p.a(object_name)} appears only in the first half of the video"
                # Shouldn't use lambda here since query_names (and number) might change.
                predicate = partial(predicate_visibility, query_names, False)
                prompt_predicate = prompt, predicate

            predicate.type = "visibility"
            predicate.texts = [f"a photo of {p.a(object_name_no_attribute)}"]
            # Limit to one box per class.
            predicate.one_box_per_class = True

            prompt_predicates += [prompt_predicate] * repeat

    return prompt_predicates


def get_prompt_predicates_1obj_dynamic_spatial(repeat=1, left_right_only=True):
    prompt_predicates = []

    # NOTE: the boxes are in (x_min, y_min, x_max, y_max) format. This is because LVD uses `condition` rather than `gen_boxes` format as the input. `condition` has processed the coordinates.
    locations = [
        (
            "left",
            "right",
            lambda box1, box2: (box1[0] + box1[2]) / 2 < (box2[0] + box2[2]) / 2,
        ),
        (
            "right",
            "left",
            lambda box1, box2: (box1[0] + box1[2]) / 2 > (box2[0] + box2[2]) / 2,
        ),
    ]
    if not left_right_only:
        # NOTE: the boxes are in (x_min, y_min, x_max, y_max) format.
        locations += [
            (
                "top",
                "bottom",
                lambda box1, box2: (box1[1] + box1[3]) / 2 < (box2[1] + box2[3]) / 2,
            ),
            (
                "bottom",
                "top",
                lambda box1, box2: (box1[1] + box1[3]) / 2 > (box2[1] + box2[3]) / 2,
            ),
        ]

    # We use object names without motion attributes for spatial since the attribute words may interfere with the intended motion.
    for object_name_no_attribute in evaluate_classes_no_attribute:
        # `query_names` needs to match with `texts` since `query_names` will be searched in the detection of `texts`
        query_names = (object_name_no_attribute,)

        for location1, location2, verify_fn in locations:
            prompt = f"{prompt_prefix} with {p.a(object_name_no_attribute)} moving from the {location1} to the {location2}"
            prompt = prompt.strip()

            predicate = partial(predicate_1obj_dynamic_spatial, query_names, verify_fn)
            prompt_predicate = prompt, predicate
            predicate.type = "dynamic_spatial"
            predicate.texts = [f"a photo of {p.a(object_name_no_attribute)}"]
            # Limit to one box per class.
            predicate.one_box_per_class = True

            prompt_predicates += [prompt_predicate] * repeat

    return prompt_predicates


def get_prompt_predicates_2obj_dynamic_spatial(
    num_prompts=10, repeat=1, left_right_only=True
):
    prompt_predicates = []

    # NOTE: the boxes are in (x_min, y_min, x_max, y_max) format. This is because LVD uses `condition` rather than `gen_boxes` format as the input. `condition` has processed the coordinates.
    locations = [
        (
            "left",
            "right",
            lambda box1, box2: (box1[0] + box1[2]) / 2 < (box2[0] + box2[2]) / 2,
        ),
        (
            "right",
            "left",
            lambda box1, box2: (box1[0] + box1[2]) / 2 > (box2[0] + box2[2]) / 2,
        ),
    ]
    if not left_right_only:
        # NOTE: the boxes are in (x_min, y_min, x_max, y_max) format.
        locations += [
            (
                "top",
                "bottom",
                lambda box1, box2: (box1[1] + box1[3]) / 2 < (box2[1] + box2[3]) / 2,
            ),
            (
                "bottom",
                "top",
                lambda box1, box2: (box1[1] + box1[3]) / 2 > (box2[1] + box2[3]) / 2,
            ),
        ]

    # We use object names without motion attributes for spatial since the attribute words may interfere with the intended motion.
    for ind in range(num_prompts):
        np.random.seed(ind)
        for location1, location2, verify_fn in locations:
            object_name1, object_name2 = np.random.choice(
                evaluate_classes_no_attribute, 2, replace=False
            )

            object_name1, query_names1 = process_object_name(object_name1)
            object_name2, query_names2 = process_object_name(object_name2)

            prompt = f"{prompt_prefix} with {p.a(object_name1)} moving from the {location1} of {p.a(object_name2)} to its {location2}"
            prompt = prompt.strip()

            # `query_names` needs to match with `texts` since `query_names` will be searched in the detection of `texts`
            predicate = partial(
                predicate_2obj_dynamic_spatial, query_names1, query_names2, verify_fn
            )
            prompt_predicate = prompt, predicate
            predicate.type = "dynamic_spatial"
            predicate.texts = [
                f"a photo of {p.a(object_name1)}",
                f"a photo of {p.a(object_name2)}",
            ]
            # Limit to one box per class.
            predicate.one_box_per_class = True
            prompt_predicates += [prompt_predicate] * repeat

    return prompt_predicates


def get_prompt_predicates_sequential(repeat=1):
    prompt_predicates = []

    locations = [
        ("lower left", "lower right", "upper right"),
        ("lower left", "upper left", "upper right"),
        ("lower right", "lower left", "upper left"),
        ("lower right", "upper right", "upper left"),
    ]
    verify_fns = {
        # lower: y is large
        "lower left": lambda box: (box[1] + box[3]) / 2 > 0.5
        and (box[0] + box[2]) / 2 < 0.5,
        "lower right": lambda box: (box[1] + box[3]) / 2 > 0.5
        and (box[0] + box[2]) / 2 > 0.5,
        "upper left": lambda box: (box[1] + box[3]) / 2 < 0.5
        and (box[0] + box[2]) / 2 < 0.5,
        "upper right": lambda box: (box[1] + box[3]) / 2 < 0.5
        and (box[0] + box[2]) / 2 > 0.5,
    }

    for object_name_no_attribute in evaluate_classes_no_attribute:
        # `query_names` needs to match with `texts` since `query_names` will be searched in the detection of `texts`
        query_names = (object_name_no_attribute,)

        for location1, location2, location3 in locations:
            # We check the appearance/disappearance in addition to whether the object is on the right side in the last frame compared to the initial frame.
            prompt = f"{prompt_top_down_prefix} in which {p.a(object_name_no_attribute)} initially on the {location1} of the scene. It first moves to the {location2} of the scene and then moves to the {location3} of the scene."

            # Shouldn't use lambda here since query_names (and number) might change.
            predicate = partial(
                predicate_sequentialv2,
                query_names,
                verify_fns[location1],
                verify_fns[location2],
                verify_fns[location3],
            )
            # predicate = partial(predicate_sequentialv2, query_names, verify_fn1, verify_fn2)
            prompt_predicate = prompt, predicate
            predicate.type = "sequential"
            predicate.texts = [f"a photo of {p.a(object_name_no_attribute)}"]
            # Limit to one box per class.
            predicate.one_box_per_class = True
            prompt_predicates += [prompt_predicate] * repeat

    return prompt_predicates


def get_lvd_full_prompt_predicates(prompt_type=None):
    # numeracy: 100 prompts, number 1 to 4, 5 classes, repeat 5 times
    prompt_predicates_numeracy = get_prompt_predicates_numeracy(max_num=4, repeat=5)
    # attribution: 100 prompts, two objects in each prompt, each with attributes (randomly sampled)
    prompt_predicates_attribution = get_prompt_predicates_attribution(num_prompts=100)
    # visibility: 100 prompts, 5 classes, appear/disappear, repeat 10 times
    prompt_predicates_visibility = get_prompt_predicates_visibility(repeat=10)
    # dynamic spatial: 100 prompts
    # 1 object: 50 prompts, 5 classes, left/right, repeat 5 times
    # 2 objects: 50 prompts, randomly sample two objects 25 times, left/right
    prompt_predicates_1obj_dynamic_spatial = get_prompt_predicates_1obj_dynamic_spatial(
        repeat=5
    )
    prompt_predicates_2obj_dynamic_spatial = get_prompt_predicates_2obj_dynamic_spatial(
        num_prompts=25
    )
    prompt_predicates_dynamic_spatial = (
        prompt_predicates_1obj_dynamic_spatial + prompt_predicates_2obj_dynamic_spatial
    )
    # sequential: 100 prompts, 5 classes, 4 location triplets, repeat 5 times
    prompt_predicates_sequential = get_prompt_predicates_sequential(repeat=5)

    prompt_predicates_static_all = (
        prompt_predicates_numeracy + prompt_predicates_attribution
    )
    prompts_predicates_dynamic_all = (
        prompt_predicates_visibility
        + prompt_predicates_dynamic_spatial
        + prompt_predicates_sequential
    )

    # Each one has 100 prompts
    prompt_predicates_all = (
        prompt_predicates_numeracy
        + prompt_predicates_attribution
        + prompt_predicates_visibility
        + prompt_predicates_dynamic_spatial
        + prompt_predicates_sequential
    )

    prompt_predicates = {
        "lvd": prompt_predicates_all,
        "lvd_static": prompt_predicates_static_all,
        "lvd_numeracy": prompt_predicates_numeracy,
        "lvd_attribution": prompt_predicates_attribution,
        "lvd_dynamic": prompts_predicates_dynamic_all,
        "lvd_dynamic_spatial": prompt_predicates_dynamic_spatial,
        "lvd_visibility": prompt_predicates_visibility,
        "lvd_sequential": prompt_predicates_sequential,
    }

    if prompt_type is not None:
        return prompt_predicates[prompt_type]
    else:
        return prompt_predicates


def get_lvd_full_prompts(prompt_type):
    prompt_predicates = get_lvd_full_prompt_predicates(prompt_type)
    if prompt_type is not None:
        return [item[0] for item in prompt_predicates]
    else:
        return {k: [item[0] for item in v] for k, v in prompt_predicates.items()}


if __name__ == "__main__":
    prompt_predicates = get_lvd_full_prompt_predicates("lvdv1.1")

    print(
        np.unique(
            [predicate.type for prompt, predicate in prompt_predicates],
            return_counts=True,
        )
    )
    print(len(prompt_predicates))
