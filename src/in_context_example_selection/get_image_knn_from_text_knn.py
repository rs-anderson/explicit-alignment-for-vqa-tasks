import pickle
from easydict import EasyDict
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import faiss
import numpy as np
from collections import defaultdict


data_dir = Path("/home/rsa39/rds/rds-mlmi-2020-21-xyBFuSj0hm0/MLMI.2021-22/rsa39/project/RAVQA/data")
vqa2_data_dir = data_dir / "vqa2"


with open("text_knns_reformatted.pkl", "rb") as f:
    text_nearest_neighbours_dict = pickle.load(f)

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



res = faiss.StandardGpuResources()
k = 2048    
results = defaultdict(dict)

for test_example in tqdm(val_data_vqa2.data_items):
    train_img_embeddings_to_search_over = []
    img_keys_for_searched_embeddings = []

    val_image_embedding = val_image_embeddings.get(str(test_example.img_key))
    question_ids_for_question_nearest_neighbours = text_nearest_neighbours_dict.get(str(test_example.question_id))["question_ids"]
    
    if not val_image_embedding:
        print(f"There is no image embedding for {str(test_example.img_key)}")
    
    if not question_ids_for_question_nearest_neighbours:
        print(f"There is no question nns for {str(test_example.question_id)}")

    for train_question_id in question_ids_for_question_nearest_neighbours:
        train_vqa_data_item = vqa2_data_by_q_id[int(train_question_id)]
        img_key = train_vqa_data_item.img_key
        if img_key not in img_keys_for_searched_embeddings:
            img_keys_for_searched_embeddings.append(img_key)
            train_img_embeddings_to_search_over.append(train_image_embeddings.get(str(img_key)))

    train_img_embeddings_to_search_over = np.stack(train_img_embeddings_to_search_over).squeeze(1)
    
    index = faiss.IndexFlatIP(train_img_embeddings_to_search_over.shape[1])   # build the index
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    faiss.normalize_L2(train_img_embeddings_to_search_over)
    gpu_index.add(train_img_embeddings_to_search_over)                  # add vectors to the index

    k = len(img_keys_for_searched_embeddings)
    faiss.normalize_L2(val_image_embedding)         
    D, I = gpu_index.search(val_image_embedding, k)
    
    results[test_example.question_id]["D"] = D
    results[test_example.question_id]["I"] = I
    results[test_example.question_id]["img_keys"] = img_keys_for_searched_embeddings

with open("results.pkl", "wb") as f:
    pickle.dump(results, f)


# reformatting the output so that I -> img_keys

output_dict = {}
for test_question_id, results_dict in tqdm(results.items()):
    temp_dict = {}
    temp_dict["similarities"] = results_dict["D"]
    temp_dict["img_keys"] = [results_dict["img_keys"][ind] for ind in results_dict["I"].flatten()]
    output_dict[test_question_id] = temp_dict


with open("image_knns_reformatted.pkl", "wb") as f:
    pickle.dump(output_dict, f)