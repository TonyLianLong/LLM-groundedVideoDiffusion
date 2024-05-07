import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import torch
from utils.llm import get_full_model_name, model_names
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from glob import glob
from utils.eval import to_gen_box_format, evaluate_with_layout, class_aware_nms, nms
from tqdm import tqdm
import numpy as np
import json
import joblib
from prompt import get_prompts, prompt_types

torch.set_grad_enabled(False)


def keep_one_box_per_class(boxes, scores, labels):
    # Keep the box with highest label per class

    boxes_output, scores_output, labels_output = [], [], []
    labels_unique = np.unique(labels)
    for label in labels_unique:
        label_mask = labels == label
        boxes_label = boxes[label_mask]
        scores_label = scores[label_mask]
        max_score_index = scores_label.argmax()
        box, score = boxes_label[max_score_index], scores_label[max_score_index]
        boxes_output.append(box)
        scores_output.append(score)
        labels_output.append(label)

    return np.array(boxes_output), np.array(scores_output), np.array(labels_output)


def eval_prompt(
    p,
    predicate,
    path,
    processor,
    model,
    score_threshold=0.1,
    nms_threshold=0.5,
    use_class_aware_nms=False,
    num_eval_frames=6,
    use_cuda=True,
    verbose=False,
):
    video = joblib.load(path)
    texts = [predicate.texts]

    parsed_layout = {"Prompt": p, "Background keyword": None}

    eval_frame_indices = (
        np.round(np.linspace(0, len(video) - 1, num_eval_frames)).astype(int).tolist()
    )

    assert len(set(eval_frame_indices)) == len(
        eval_frame_indices
    ), f"Eval indices not unique: {eval_frame_indices}"

    print(f"Eval indices: {eval_frame_indices}")

    frame_ind = 1
    for eval_frame_index in eval_frame_indices:
        image = video[eval_frame_index]
        inputs = processor(text=texts, images=image, return_tensors="pt")
        if use_cuda:
            inputs = inputs.to("cuda")
        outputs = model(**inputs)

        height, width, _ = image.shape

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([[height, width]])
        if use_cuda:
            target_sizes = target_sizes.cuda()
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        text = texts[i]
        boxes, scores, labels = (
            results[i]["boxes"],
            results[i]["scores"],
            results[i]["labels"],
        )
        boxes = boxes.cpu()
        # xyxy ranging from 0 to 1
        boxes = np.array(
            [
                [x_min / width, y_min / height, x_max / width, y_max / height]
                for (x_min, y_min, x_max, y_max), score in zip(boxes, scores)
                if score >= score_threshold
            ]
        )
        labels = np.array(
            [
                label.cpu().numpy()
                for label, score in zip(labels, scores)
                if score >= score_threshold
            ]
        )
        scores = np.array(
            [score.cpu().numpy() for score in scores if score >= score_threshold]
        )

        # print(f"Pre-NMS:")
        # for box, score, label in zip(boxes, scores, labels):
        #     box = [round(i, 2) for i in box.tolist()]
        #     print(
        #         f"Detected {text[label]} ({label}) with confidence {round(score.item(), 3)} at location {box}")

        print(f"Post-NMS (frame frame_ind):")

        if use_class_aware_nms:
            boxes, scores, labels = class_aware_nms(
                boxes, scores, labels, nms_threshold
            )
        else:
            boxes, scores, labels = nms(boxes, scores, labels, nms_threshold)

        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            print(
                f"Detected {text[label]} ({label}) with confidence {round(score.item(), 3)} at location {box}"
            )

        if verbose:
            print(
                f"prompt: {p}, texts: {texts}, boxes: {boxes}, labels: {labels}, scores: {scores}"
            )

        # Here we are not using a tracker so the box id could mismatch when we have multiple objects with the same label.
        # For numeracy, we do not need tracking (mismatch is ok). For other tasks, we only include the box with max confidence.
        if predicate.one_box_per_class:
            boxes, scores, labels = keep_one_box_per_class(boxes, scores, labels)

            for box, score, label in zip(boxes, scores, labels):
                box = [round(i, 2) for i in box.tolist()]
                print(
                    f"After selection one box per class: Detected {text[label]} ({label}) with confidence {round(score.item(), 3)} at location {box}"
                )

        det_boxes = []
        label_counts = {}

        # This ensures boxes of different labels will not be matched to each other.
        for box, score, label in zip(boxes, scores, labels):
            if label not in label_counts:
                label_counts[label] = 0
            # Here we convert to gen box format (same as LLM output), xywh (in pixels). This is for compatibility with first stage evaluation. This will be converted to condition format (xyxy, ranging from 0 to 1).
            det_boxes.append(
                {
                    "id": label * 100 + label_counts[label],
                    "name": text[label],
                    "box": to_gen_box_format(box, width, height, rounding=True),
                    "score": score,
                }
            )
            label_counts[label] += 1

        parsed_layout[f"Frame {frame_ind}"] = det_boxes

        frame_ind += 1

    print(f"parsed_layout: {parsed_layout}")

    eval_type, eval_success = evaluate_with_layout(
        parsed_layout,
        predicate,
        num_parsed_layout_frames=num_eval_frames,
        height=height,
        width=width,
        verbose=verbose,
    )

    return eval_type, eval_success


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-type", type=str, default="lvd")
    parser.add_argument("--run_base_path", type=str)
    parser.add_argument("--run_start_ind", default=0, type=int)
    parser.add_argument("--num_prompts", default=None, type=int)
    parser.add_argument("--num_eval_frames", default=6, type=int)
    parser.add_argument("--skip_first_prompts", default=0, type=int)
    parser.add_argument("--detection_score_threshold", default=0.05, type=float)
    parser.add_argument("--nms_threshold", default=0.5, type=float)
    parser.add_argument("--class-aware-nms", action="store_true")
    parser.add_argument("--save-eval", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    np.set_printoptions(precision=2)

    prompt_predicates = get_prompts(args.prompt_type, return_predicates=True)
    num_eval_frames = args.num_eval_frames

    print(f"Number of prompts (predicates): {len(prompt_predicates)}")
    print(f"Number of evaluating frames: {num_eval_frames}")

    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    owl_vit_model = OwlViTForObjectDetection.from_pretrained(
        "google/owlvit-base-patch32"
    )
    owl_vit_model.eval()

    use_cuda = not args.no_cuda

    if use_cuda:
        owl_vit_model.cuda()

    eval_success_counts = {}
    eval_all_counts = {}

    eval_successes = {}

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

        search_path = f"{args.run_base_path}/{ind+args.run_start_ind}/video_*.joblib"

        # NOTE: sorted with string type
        path = sorted(glob(search_path))
        if len(path) == 0:
            print(f"***No image matching {search_path}, skipping***")
            continue
        elif len(path) > 1:
            print(f"***More than one images match {search_path}: {path}, skipping***")
            continue
        path = path[0]
        print(f"Video path: {path} ({path.replace('.joblib', '.gif')})")

        eval_type, eval_success = eval_prompt(
            prompt,
            predicate,
            path,
            processor,
            owl_vit_model,
            score_threshold=args.detection_score_threshold,
            nms_threshold=args.nms_threshold,
            use_class_aware_nms=args.class_aware_nms,
            num_eval_frames=num_eval_frames,
            use_cuda=use_cuda,
            verbose=args.verbose,
        )

        print(f"Eval success (eval_type):", eval_success)

        if eval_type not in eval_all_counts:
            eval_success_counts[eval_type] = 0
            eval_all_counts[eval_type] = 0
            eval_successes[eval_type] = []

        eval_success_counts[eval_type] += int(eval_success)
        eval_all_counts[eval_type] += 1
        eval_successes[eval_type].append(bool(eval_success))

    summary = []
    eval_success_conut, eval_all_count = 0, 0
    for k, v in eval_all_counts.items():
        rate = eval_success_counts[k] / eval_all_counts[k]
        print(
            f"Eval type: {k}, success: {eval_success_counts[k]}/{eval_all_counts[k]}, rate: {round(rate, 2):.2f}"
        )
        eval_success_conut += eval_success_counts[k]
        eval_all_count += eval_all_counts[k]
        summary.append(rate)

    rate = eval_success_conut / eval_all_count
    print(f"Overall: success: {eval_success_conut}/{eval_all_count}, rate: {rate:.2f}")
    summary.append(rate)

    summary_str = "/".join([f"{round(rate, 2):.2f}" for rate in summary])
    print(f"Summary: {summary_str}")

    if args.save_eval:
        save_eval = {}
        save_eval["success_counts"] = eval_success_counts
        save_eval["sample_counts"] = eval_all_counts
        save_eval["successes"] = eval_successes
        save_eval["success_counts_overall"] = eval_success_conut
        save_eval["sample_counts_overall"] = eval_all_count

        # Reference: https://stackoverflow.com/questions/58408054/typeerror-object-of-type-bool-is-not-json-serializable

        with open(f"{args.run_base_path}/eval.json", "w") as f:
            json.dump(save_eval, f, indent=4)
