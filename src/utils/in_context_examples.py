from collections import defaultdict
import json
import pickle
import random
from typing import List, Optional
from easydict import EasyDict

# from vqa_tools import VQA

from pathlib import Path

from sklearn.metrics import pairwise_distances
import numpy as np
from tqdm import tqdm
import clip

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

data_dir = Path("../../data")
vqa2_data_dir = data_dir / "vqa2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class EmbeddingsDataset(Dataset):

    def __init__(self, vqa2_data_by_q_id, train_image_embeddings, train_text_embeddings):
        self.vqa2_data_by_q_id = vqa2_data_by_q_id
        self.train_image_embeddings = train_image_embeddings
        self.train_text_embeddings = train_text_embeddings
        self.text_embeddings_list = list(train_text_embeddings.items())

    def __len__(self):
        return len(self.text_embeddings_list)

    def __getitem__(self, idx):
        question_id, text_embedding = self.text_embeddings_list[idx]
        vqa_entry = self.vqa2_data_by_q_id.get(int(question_id), None)
        img_key = vqa_entry["img_key"]
        image_embdding = self.train_image_embeddings.get(str(img_key), None)
        return question_id, img_key, image_embdding, text_embedding

    def collate_fn(self, batch):
        
        question_ids = []
        img_keys = []
        train_image_embeddings = []
        train_text_embeddings = []

        for question_id, img_key, image_embedding, text_embedding in batch:
            question_ids.append(question_id)
            img_keys.append(img_key)
            train_image_embeddings.append(torch.tensor(image_embedding))
            train_text_embeddings.append(torch.tensor(text_embedding))

        train_image_embeddings = torch.stack(train_image_embeddings)
        train_text_embeddings = torch.stack(train_text_embeddings)
        return question_ids, img_keys, train_image_embeddings.to(device), train_text_embeddings.to(device)


class InContextExampleSelector:
    def __init__(
        self,
        num_in_context_examples: int,
        question_ids: list,
        vqa2_data,
        train_image_embeddings=None,
        train_text_embeddings=None,
        val_image_embeddings=None,
        val_text_embeddings=None,
        clip_model=None,
    ) -> None:
        self.num_in_context_examples = num_in_context_examples
        self.question_ids = question_ids
        print("reformatting: setting question_id as key for vqa data")
        self.vqa2_data_by_q_id = {
            vqa2_data_item["question_id"]: vqa2_data_item
            for vqa2_data_item in vqa2_data
        }
        embeddings_dataset = EmbeddingsDataset(self.vqa2_data_by_q_id, train_image_embeddings=train_image_embeddings, train_text_embeddings=train_text_embeddings)
        self.embeddings_dataloader = DataLoader(embeddings_dataset, batch_size=2**8, shuffle=False, collate_fn=embeddings_dataset.collate_fn)
        self.clip_model = clip_model
        self.val_image_embeddings = val_image_embeddings
        self.val_text_embeddings = val_text_embeddings

    def get_random_examples(self):
        in_context_examples_idxs = np.random.choice(
            self.question_ids, size=self.num_in_context_examples, replace=False
        )
        in_context_examples = self._get_examples_from_ids(in_context_examples_idxs)
        return in_context_examples


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

    image_token = "<extra_id_{}>"
    formats = dict(
        default="{image_token}\n{question}\n{answer}",
        frozen="{image_token}\nQuestion: {question}\nAnswer: {answer}",
        hotpotqa="{image_token}\nCombine facts and answer this:\n{question}\n{answer}",
        extractive="Extract the answer to the question from the following context.\nQuestion: {question}\nContext: {image_token}",

        squad="Answer the question depending on the context.\nContext: {image_token};\nQuestion: {question};\nAnswer: {answer}",
        plain="{question}\nThe answer is\n{answer}",

        default_no_prefix="{question}\n{answer}",
        frozen_no_prefix="Question: {question}\nAnswer: {answer}",
        hotpotqa_no_prefix="Combine facts and answer this:\n{question}\n{answer}",
        squad_no_prefix="Answer the question depending on the context.\nContext: ;\nQuestion: {question};\nAnswer: {answer}",
        hotpotqa_list = [
            "{image_token}\nCombine facts and answer this:\n{question}\n{answer}",
            "{image_token}\nFormulate an answer to this elaborate question:\n{question}\n{answer}",
            "{image_token}\nHere's a complex question that requires someone to reason about the input, can you answer it?\n{question}\n{answer}",
        ]
    )


    def __init__(self, format_type: str, sep_token: str = "\n", pass_examples_through_encoder_one_at_a_time: Optional[bool] = False, sample_templates: Optional[bool] = False, ensemble_one_shots: Optional[bool] = False) -> None:
        self.format_type = format_type
        self.sep_token = sep_token
        self.pass_examples_through_encoder_one_at_a_time = pass_examples_through_encoder_one_at_a_time
        self.sample_templates = sample_templates
        self.ensemble_one_shots = ensemble_one_shots
        if self.sample_templates:
            format_type = format_type + "_list"
            self.input_format_list = InContextExampleFormatter.formats[format_type]
        else:
            self.input_format = InContextExampleFormatter.formats[format_type]

    def format_input(self, in_context_examples: List[EasyDict], test_example: EasyDict):
        if self.sample_templates:
            self.input_format = random.choice(self.input_format_list)

        if self.ensemble_one_shots:
            return [self.format_input_with_prefix([in_context_example], test_example) for in_context_example in in_context_examples]

        if self.format_type in ["default", "frozen", "hotpotqa", "squad", "extractive", "hotpotqa_list"]:
            return self.format_input_with_prefix(in_context_examples, test_example)
        
        else:
            return self.format_input_without_prefix(in_context_examples, test_example)

    def format_input_with_prefix(self, in_context_examples: List[EasyDict], test_example: EasyDict):
        num_in_context_examples = len(in_context_examples)
        formatted_input_list = self.format_in_context_examples(in_context_examples)
        formatted_input_list.append(self.format_test_input(num_in_context_examples, test_example))

        if self.pass_examples_through_encoder_one_at_a_time:
            return formatted_input_list
        else:
            formatted_input = self.sep_token.join(formatted_input_list)
        return formatted_input
    
    def format_input_without_prefix(self, in_context_examples: List[EasyDict], test_example: EasyDict):
        formatted_input_list = [
            self.input_format.format(
                question=example.question,
                answer=example.gold_answer + ".",
            )
            for example in in_context_examples
        ]
        formatted_input_list.append(
            self.input_format.format(
                question=test_example.question,
                answer="",
            )
        )
        if self.pass_examples_through_encoder_one_at_a_time:
            return formatted_input_list
        elif self.ensemble_one_shots:
            formatted_input = self.sep_token.join(formatted_input_list)
        else:
            formatted_input = self.sep_token.join(formatted_input_list)
        return formatted_input


    def format_in_context_examples(self, in_context_examples: List[EasyDict]):
        formatted_input_list = [
            self.input_format.format(
                image_token=InContextExampleFormatter.image_token.format(i),
                question=example.question,
                answer=example.gold_answer,
            )
            for i, example in enumerate(in_context_examples)
        ]   
        return formatted_input_list

    def format_test_input(self, num_in_context_examples: int, test_example: EasyDict):

        return (
            self.input_format.format(
                image_token=InContextExampleFormatter.image_token.format(
                    num_in_context_examples
                ),
                question=test_example.question,
                answer="",
            )
        )


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

    print("Reading train VQA2 data")
    with open(vqa2_data_dir / "cache/train_data_preprocessed.pkl", "rb") as f:
        load_pickle_data = pickle.load(f)["cache"]
    data_vqa2 = EasyDict(load_pickle_data)

    print("Reading val VQA2 data")
    with open(vqa2_data_dir / "cache/val_data_preprocessed.pkl", "rb") as f:
        load_pickle_data = pickle.load(f)["cache"]
    val_data_vqa2 = EasyDict(load_pickle_data)

    print("Reading train image embeddings")
    with open(
        vqa2_data_dir
        / "pre-extracted_features/clip_embeddings/coco_ViT-L_14@336px_train2014.pkl",
        "rb",
    ) as f:
        load_pickle_data = pickle.load(f)
    train_image_embeddings = EasyDict(load_pickle_data)

    print("Reading val image embeddings")
    with open(
        vqa2_data_dir
        / "pre-extracted_features/clip_embeddings/coco_ViT-L_14@336px_val2014.pkl",
        "rb",
    ) as f:
        load_pickle_data = pickle.load(f)
    val_image_embeddings = EasyDict(load_pickle_data)

    print("Reading train text embeddings")
    with open(
        vqa2_data_dir
        / "pre-extracted_features/text_embeddings/coco_ViT-L_14@336px_train2014.pkl",
        "rb",
    ) as f:
        load_pickle_data = pickle.load(f)
    train_text_embeddings = EasyDict(load_pickle_data)


    print("Reading val text embeddings")
    with open(
        vqa2_data_dir
        / "pre-extracted_features/text_embeddings/coco_ViT-L_14@336px_val2014.pkl",
        "rb",
    ) as f:
        load_pickle_data = pickle.load(f)
    val_text_embeddings = EasyDict(load_pickle_data)

    np.random.seed(2021)

    random_example_selector = InContextExampleSelector(
            num_in_context_examples=16,
            question_ids=train_question_ids,
            vqa2_data=data_vqa2.data_items,
    )
    

    random_in_context_examples_for_val_set = {
        str(question_id): random_example_selector.get_random_examples()
        for question_id in tqdm(val_question_ids)
    }

    print(len(random_in_context_examples_for_val_set))

    out_path = vqa2_data_dir / f"pre-extracted_features/in_context_examples/random.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(random_in_context_examples_for_val_set, f)

