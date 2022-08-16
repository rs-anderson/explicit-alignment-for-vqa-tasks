import argparse
import pickle
from pathlib import Path
from re import I
from typing import Dict, List, Union
from easydict import EasyDict
import matplotlib.pyplot as plt
import skimage.io as io

data_dir = Path("../../data")
okvqa_data_dir = data_dir / "ok-vqa"
vqa2_data_dir = data_dir / "vqa2"
in_context_examples_dir = vqa2_data_dir / "pre-extracted_features" / "in_context_examples"
cached_val_vqa_data_path = vqa2_data_dir / "cache/val_data_preprocessed.pkl"


def main(test_question_id: str, in_context_example_fname: str, num_in_context_examples: int):
    
    in_context_examples = load_in_context_examples_for_question_id(in_context_example_fname, test_question_id)
    in_context_examples = select_best_k(in_context_examples, num_in_context_examples)
    test_vqa_example = load_vqa_item_for_question_id(test_question_id)
    test_question_id, test_img_key, test_question, test_gold_answer = unpack_example_items(**test_vqa_example)

    fig, axs = plt.subplots(num_in_context_examples + 1, figsize=(3*num_in_context_examples, 3*num_in_context_examples))
    for i, in_context_example in enumerate(in_context_examples[-num_in_context_examples:]):
        _, img_key, question, gold_answer = unpack_example_items(**in_context_example)
        image = load_image(img_key, "train2014")
        axs[i].imshow(image)
        axs[i].set_title(f"{question}")
        axs[i].axis('off')
    
    test_image = load_image(test_img_key, "val2014")
    axs[-1].imshow(test_image)
    axs[-1].set_title(f"{test_question}", color="navy")
    axs[-1].axis('off')
    
    fig.savefig(in_context_examples_dir / f"{test_question_id}_{in_context_example_fname.split('.')[0]}_{num_in_context_examples}.png", bbox_inches='tight') 



def load_in_context_examples_for_question_id(in_context_example_fname: str, test_question_id: str):
    in_context_examples_file_path = in_context_examples_dir / in_context_example_fname

    print(f"Reading in-context examples from {in_context_examples_file_path}")
    with open(in_context_examples_file_path, "rb") as f:
        in_context_examples = pickle.load(f)
    try:
        in_context_examples = in_context_examples[test_question_id]
    except KeyError:
        print(f"'{test_question_id}' not in in_context_examples")
        print(f"Trying {test_question_id} as an int")
        try:
            in_context_examples = in_context_examples[int(test_question_id)]
        except KeyError:
            print("Also not in in-context examples")
            exit(0)
    return in_context_examples


def load_vqa_item_for_question_id(test_question_id: str):
    print(f"Reading val VQA2 data from {cached_val_vqa_data_path}")
    with open(cached_val_vqa_data_path, "rb") as f:
        load_pickle_data = pickle.load(f)["cache"]
    
    val_data_vqa2 = EasyDict(load_pickle_data)
    val_vqa2_data_by_q_id = {
        vqa2_data_item["question_id"]: vqa2_data_item
        for vqa2_data_item in val_data_vqa2.data_items
    }

    return val_vqa2_data_by_q_id[int(test_question_id)]


def select_best_k(in_context_examples: List, num_in_context_examples: int):
    print(f"Keeping the {num_in_context_examples} best in-context examples")
    return in_context_examples[-num_in_context_examples:]


def load_image(img_key: str, subtype: str):
    img_path = (
        okvqa_data_dir / f"{subtype}/COCO_{subtype}_{int(img_key):012d}.jpg"
    )
    image = io.imread(img_path)
    return image


def unpack_example_items(question_id: str, img_key: int, question: str, gold_answer: str, *args, **kwargs):
    return question_id, str(img_key), question, gold_answer


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--question_id",
        type=str,
        help="A question id from the validation set.",
    )
    arg_parser.add_argument(
        "--in_context_examples_fname",
        type=str,
        help="File name of the in-context example file.",
    )
    arg_parser.add_argument(
        "--num_in_context_examples",
        type=int,
        help="Number of in-context examples.",
    )
    args = arg_parser.parse_args()

    main(args.question_id, args.in_context_examples_fname, args.num_in_context_examples)
