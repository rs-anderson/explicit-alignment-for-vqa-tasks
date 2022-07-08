from collections import defaultdict
import json
import pickle
from typing import List
from easydict import EasyDict

# from vqa_tools import VQA

from pathlib import Path

from sklearn.metrics import pairwise_distances
import numpy as np
from tqdm import tqdm

data_dir = Path("../../data")
vqa2_data_dir = data_dir / "vqa2"


class InContextExampleSelector:
    def __init__(
        self,
        num_in_context_examples: int,
        question_ids: list,
        vqa2_data,
        image_embeddings=None,
        text_embeddings=None,
    ) -> None:
        self.num_in_context_examples = num_in_context_examples
        self.image_embeddings = image_embeddings
        self.text_embeddings = text_embeddings
        self.question_ids = question_ids
        print("reformatting: setting question_id as key for vqa data")
        self.vqa2_data_by_q_id = {
            vqa2_data_item["question_id"]: vqa2_data_item
            for vqa2_data_item in vqa2_data
        }

    def get_random_examples(self):
        in_context_examples_idxs = np.random.choice(
            self.question_ids, size=self.num_in_context_examples, replace=False
        )
        in_context_examples = self._get_examples_from_ids(in_context_examples_idxs)
        return in_context_examples

    def get_rice_examples(self, example):
        pass

    def _get_examples_from_ids(self, in_context_examples_idxs):
        in_context_examples = []
        for idx in in_context_examples_idxs:
            data_item = self.vqa2_data_by_q_id[idx]
            question_id = data_item["question_id"]
            img_key = data_item["img_key"]
            in_context_examples.append(
                {
                    "question_id": question_id,
                    "img_key": img_key,
                    "question": data_item["question"],
                    "gold_answer": data_item["gold_answer"],
                }
            )
        return in_context_examples


class InContextExampleFormatter:
    # can apply permutation here if necessary

    image_token = "<extra_id_{}>"
    formats = dict(
        frozen="{image_token}\n{question}\n{answer}",
        hotpotqa="{image_token}\nCombine facts and answer this:\n{question}\n{answer}",
    )

    def __init__(self, format_type: str, sep_token: str = "\n") -> None:
        self.format_type = format_type
        self.sep_token = sep_token
        self.input_format = InContextExampleFormatter.formats[format_type]

    # TODO: refactor to two methods: one for each type of input (i.e. in-context + test example).
    def format_input(self, in_context_examples: List[EasyDict], test_example: EasyDict):
        num_in_context_examples = len(in_context_examples)
        formatted_input_list = [
            self.input_format.format(
                image_token=InContextExampleFormatter.image_token.format(i),
                question=example.question,
                answer=example.gold_answer,
            )
            for i, example in enumerate(in_context_examples)
        ]
        formatted_input_list.append(
            self.input_format.format(
                image_token=InContextExampleFormatter.image_token.format(
                    num_in_context_examples
                ),
                question=test_example.question,
                answer="",
            )
        )
        formatted_input = self.sep_token.join(formatted_input_list)
        return formatted_input



if __name__ == "__main__":

    with open(vqa2_data_dir / "v2_OpenEnded_mscoco_train2014_questions.json", "r") as f:
        data_items_list = json.load(f)["questions"]

    print("%0d questions loaded from json " % len(data_items_list))
    train_question_ids = [
        data_item_dict["question_id"] for data_item_dict in data_items_list
    ]

    with open(vqa2_data_dir / "v2_OpenEnded_mscoco_val2014_questions.json", "r") as f:
        data_items_list = json.load(f)["questions"]

    print("%0d questions loaded from json " % len(data_items_list))
    val_question_ids = [
        data_item_dict["question_id"] for data_item_dict in data_items_list
    ]

    print("Reading VQA2 data")
    with open(vqa2_data_dir / "cache/train_data_preprocessed.pkl", "rb") as f:
        load_pickle_data = pickle.load(f)["cache"]
    data_vqa2 = EasyDict(load_pickle_data)

    print("Reading image embeddings")
    with open(
        vqa2_data_dir
        / "pre-extracted_features/clip_embeddings/coco_ViT-L_14@336px_train2014.pkl",
        "rb",
    ) as f:
        load_pickle_data = pickle.load(f)
    image_embeddings = EasyDict(load_pickle_data)

    np.random.seed(2021)

    example_selector = InContextExampleSelector(
            num_in_context_examples=64,
            question_ids=train_question_ids,
            vqa2_data=data_vqa2.data_items,
            image_embeddings=image_embeddings,
        )
    
    for i in range(10):

        random_in_context_examples_for_val_set = {}

        random_in_context_examples_for_val_set = {
            str(question_id): example_selector.get_random_examples()
            for question_id in tqdm(val_question_ids)
        }

        print(len(random_in_context_examples_for_val_set))

        out_path = vqa2_data_dir / f"pre-extracted_features/in_context_examples/random_{i}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(random_in_context_examples_for_val_set, f)
