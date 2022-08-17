import pickle
from easydict import EasyDict
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import faiss
import numpy as np

data_dir = Path("/home/rsa39/rds/rds-mlmi-2020-21-xyBFuSj0hm0/MLMI.2021-22/rsa39/project/RAVQA/data")
vqa2_data_dir = data_dir / "vqa2"


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


train_text_embeddings_dict = dict(
    question_ids=[],
    text_embeddings=[],
)

for question_id, text_embedding in tqdm(train_text_embeddings.items()):
    train_text_embeddings_dict["question_ids"].append(question_id)
    train_text_embeddings_dict["text_embeddings"].append(text_embedding)

train_text_embeddings_df = pd.DataFrame.from_dict(train_text_embeddings_dict)  # for mapping faiss index to img key
train_text_embeddings_database = np.stack(train_text_embeddings_dict["text_embeddings"]).squeeze(1)  # for faiss index database

print(train_text_embeddings_df.head())
print(train_text_embeddings_database.shape)


val_text_embeddings_dict = dict(
    question_ids=[],
    text_embeddings=[],
)

for question_id, text_embedding in tqdm(val_text_embeddings.items()):
    val_text_embeddings_dict["question_ids"].append(question_id)
    val_text_embeddings_dict["text_embeddings"].append(text_embedding)

val_text_embeddings_df = pd.DataFrame.from_dict(val_text_embeddings_dict)  # for mapping faiss index to img key
val_text_embeddings_database = np.stack(val_text_embeddings_dict["text_embeddings"]).squeeze(1)  # for faiss index database

print(val_text_embeddings_df.head())
print(val_text_embeddings_database.shape)

res = faiss.StandardGpuResources()

index = faiss.IndexFlatIP(train_text_embeddings_database.shape[1])   # build the index
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
print(gpu_index.is_trained)
faiss.normalize_L2(train_text_embeddings_database)
gpu_index.add(train_text_embeddings_database)                  # add vectors to the gpu_index
print(gpu_index.ntotal)


faiss.normalize_L2(val_text_embeddings_database)
k = 2048                          # we want to see 4 nearest neighbors
D, I = gpu_index.search(val_text_embeddings_database, k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I.shape)


with open('text_nearest_neighbours_2048.npy', 'wb') as f:
    np.save(f, I)
with open('text_nearest_neighbours_similarities_2048.npy', 'wb') as f:
    np.save(f, D)