import pickle
from easydict import EasyDict
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np

data_dir = Path("/home/rsa39/rds/rds-mlmi-2020-21-xyBFuSj0hm0/MLMI.2021-22/rsa39/project/RAVQA/data")
vqa2_data_dir = data_dir / "vqa2"

text_nearest_neighbours = np.load("text_nearest_neighbours_2048.npy")
text_nearest_neighbours_similarities = np.load("text_nearest_neighbours_similarities_2048.npy")

train_text_embeddings_df = pd.read_pickle("train_text_embeddings_df.pkl")
val_text_embeddings_df = pd.read_pickle("val_text_embeddings_df.pkl")

text_nearest_neighbours_dict = dict()
for i in tqdm(range(text_nearest_neighbours.shape[0])):
    temp_dict = {}
    val_question_id = val_text_embeddings_df.iloc[i]["question_ids"]
    temp_dict["question_ids"] = train_text_embeddings_df.loc[text_nearest_neighbours[i]]["question_ids"].values
    temp_dict["similarities"] = text_nearest_neighbours_similarities[i]
    text_nearest_neighbours_dict[val_question_id] = temp_dict

with open('text_knns_reformatted.pkl', 'wb') as f:
    pickle.dump(text_nearest_neighbours_dict, f)