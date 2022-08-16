import torch
import pickle
import json
from tqdm import tqdm
import argparse
from pathlib import Path
import clip
import numpy as np

data_dir = Path("../../data")
vqa2_data_dir = data_dir / "vqa2"
okvqa_data_dir = data_dir / "ok-vqa"


def main(clip_model_name: str, subtype: str = "val2014"):

    print(f"Extracting {subtype} using {clip_model_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(clip_model_name, device=device)
    
    clip_model_name = clip_model_name.replace('/', '_')
    
    out_path = (
        vqa2_data_dir
        / "pre-extracted_features"
        / "text_embeddings"
        / f"coco_{clip_model_name}_{subtype}.pkl"
    )

    with open(
        vqa2_data_dir / f"v2_OpenEnded_mscoco_{subtype}_questions.json", "r"
    ) as f:
        data = json.load(f)

    assert data["data_subtype"] == subtype, "wrong dataset subtype was loaded"
    print(data["info"])

    questions = data["questions"]
    print("%0d questions loaded from json " % len(questions))

    question_ids_with_embeddings = {}

    for i in tqdm(range(len(questions))):
        data_item = questions[i]

        # img_id = str(data_item["image_id"])
        question_id = str(data_item["question_id"])
        
        if question_id in question_ids_with_embeddings:
            print("Already seen question id? Strange...")
            continue
        
        tokenized_question = clip.tokenize(data_item['question']).to(device)

        with torch.no_grad():
            # prefix = model.get_image_features(image).numpy()
            text_embedding = model.encode_text(tokenized_question).cpu().numpy().astype(np.float32)

        question_ids_with_embeddings[question_id] = text_embedding

        if (i + 1) % 10000 == 0:
            with open(out_path, "wb") as f:
                pickle.dump(question_ids_with_embeddings, f)

    with open(out_path, "wb") as f:
        pickle.dump(question_ids_with_embeddings, f)

    print("Done")
    print("%0d embeddings saved " % len(question_ids_with_embeddings))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip_model_type",
        default="ViT-L/14@336px",
        # choices=("clip-vit-base-patch32"),
    )
    parser.add_argument(
        "--split", default="train2014", choices=("train2014", "val2014")
    )
    args = parser.parse_args()
    main(args.clip_model_type, args.split)
