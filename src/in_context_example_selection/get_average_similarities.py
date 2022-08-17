import pickle
from easydict import EasyDict
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np

data_dir = Path("/home/rsa39/rds/rds-mlmi-2020-21-xyBFuSj0hm0/MLMI.2021-22/rsa39/project/RAVQA/data")
vqa2_data_dir = data_dir / "vqa2"

image_nns_path = "/home/rsa39/rds/rds-mlmi-2020-21-xyBFuSj0hm0/MLMI.2021-22/rsa39/project/faiss/image_knns_reformatted.pkl"
question_nns_path = "/home/rsa39/rds/rds-mlmi-2020-21-xyBFuSj0hm0/MLMI.2021-22/rsa39/project/faiss/text_knns_reformatted.pkl"


if __name__ == "__main__":

    rices_for_image_and_question = False
    rices_for_question_only = True
    
    print("Reading image nearest neighbours")
    with open(image_nns_path, "rb") as f:
        image_nns = pickle.load(f)

    print("Reading question nearest neighbours")
    with open(question_nns_path, "rb") as f:
        question_nns = pickle.load(f)

    print("Reading train VQA2 data")
    with open(vqa2_data_dir / "cache/train_data_preprocessed.pkl", "rb") as f:
        load_pickle_data = pickle.load(f)["cache"]
    data_vqa2 = EasyDict(load_pickle_data)

    print("Reading val VQA2 data")
    with open(vqa2_data_dir / "cache/val_data_preprocessed.pkl", "rb") as f:
        load_pickle_data = pickle.load(f)["cache"]
    val_data_vqa2 = EasyDict(load_pickle_data)

    vqa2_data_by_q_id = {
            vqa2_data_item["question_id"]: vqa2_data_item
            for vqa2_data_item in data_vqa2.data_items
    }

    def img_key_from_question_id(question_id):
        return vqa2_data_by_q_id[int(question_id)].img_key

    def get_in_context_examples_for_question_id(test_question_id):

        most_similar_question_ids_and_similarities = question_nns.get(str(test_question_id))
        most_similar_img_keys_and_similarities = image_nns.get(test_question_id)

        image_nns_dict_with_df = pd.DataFrame({"img_keys": most_similar_img_keys_and_similarities["img_keys"], "similarities": most_similar_img_keys_and_similarities["similarities"].flatten()})
        question_nns_dict_with_df = pd.DataFrame({"question_ids": most_similar_question_ids_and_similarities["question_ids"], "similarities": most_similar_question_ids_and_similarities["similarities"].flatten()})

        question_nns_dict_with_df["img_keys"] = question_nns_dict_with_df["question_ids"].apply(img_key_from_question_id)

        df = pd.merge(image_nns_dict_with_df, question_nns_dict_with_df, how="inner", on="img_keys", suffixes=("_images", "_questions"))
        df['joint_similarities'] = df["similarities_images"] + df["similarities_questions"]
        topk_most_similar_examples = df.nlargest(32, "joint_similarities", keep="all").sort_values(by="joint_similarities")[["joint_similarities", "img_keys", "question_ids"]]

        in_context_examples = []
        for _, (_, img_key, question_id) in topk_most_similar_examples.iterrows():
            vqa_entry = vqa2_data_by_q_id.get(int(question_id), None)
            in_context_examples.append(
                {
                    "question_id": question_id,
                    "img_key": img_key,
                    "question": vqa_entry["question"],
                    "gold_answer": vqa_entry["gold_answer"],
                }
            )
        return in_context_examples

    def get_question_only_in_context_examples_for_question_id(test_question_id):

        most_similar_question_ids_and_similarities = question_nns.get(str(test_question_id))

        question_nns_dict_with_df = pd.DataFrame({"question_ids": most_similar_question_ids_and_similarities["question_ids"], "similarities": most_similar_question_ids_and_similarities["similarities"].flatten()})
        question_nns_dict_with_df["img_keys"] = question_nns_dict_with_df["question_ids"].apply(img_key_from_question_id)

        topk_most_similar_examples = question_nns_dict_with_df.nlargest(32, "similarities", keep="all").sort_values(by="similarities")[["similarities", "img_keys", "question_ids"]]

        in_context_examples = []
        for _, (_, img_key, question_id) in topk_most_similar_examples.iterrows():
            vqa_entry = vqa2_data_by_q_id.get(int(question_id), None)
            in_context_examples.append(
                {
                    "question_id": question_id,
                    "img_key": img_key,
                    "question": vqa_entry["question"],
                    "gold_answer": vqa_entry["gold_answer"],
                }
            )
        return in_context_examples

    if rices_for_image_and_question:
        # rices when using image and question
        rices_in_context_examples_for_val_set = {
            str(vqa_data_item["question_id"]): get_in_context_examples_for_question_id(vqa_data_item["question_id"])
            for vqa_data_item in tqdm(val_data_vqa2.data_items)
        }


        print(len(rices_in_context_examples_for_val_set))

        out_path = vqa2_data_dir / f"pre-extracted_features/in_context_examples/rices.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(rices_in_context_examples_for_val_set, f)
    
    elif rices_for_question_only:

        rices_in_context_examples_for_val_set = {
            str(vqa_data_item["question_id"]): get_question_only_in_context_examples_for_question_id(vqa_data_item["question_id"])
            for vqa_data_item in tqdm(val_data_vqa2.data_items)
        }


        print(len(rices_in_context_examples_for_val_set))

        out_path = vqa2_data_dir / f"pre-extracted_features/in_context_examples/rices_questions_only.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(rices_in_context_examples_for_val_set, f)
