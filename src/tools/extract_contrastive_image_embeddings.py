import torch
import skimage.io as io
from transformers import CLIPFeatureExtractor, CLIPModel
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
from pathlib import Path
import cv2
import clip
import numpy as np

data_dir = Path("../../data")
vqa2_data_dir = data_dir / "vqa2"
okvqa_data_dir = data_dir / "ok-vqa"


def main(clip_model_name: str, subtype: str = "val2014"):

    print(f"Extracting {subtype} using {clip_model_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, image_preprocessor = clip.load(clip_model_name, device=device)
    
    clip_model_name = clip_model_name.replace('/', '_')
    
    out_path = (
        vqa2_data_dir
        / "pre-extracted_features"
        / "clip_embeddings"
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

    img_ids_with_embeddings = {}

    for i in tqdm(range(len(questions))):
        data_item = questions[i]

        img_id = str(data_item["image_id"])

        if img_id in img_ids_with_embeddings:
            continue

        img_path = (
            okvqa_data_dir / f"{subtype}/COCO_{subtype}_{int(img_id):012d}.jpg"
        )
        image = io.imread(img_path)

        image = image_preprocessor(Image.fromarray(image)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # prefix = model.get_image_features(image).numpy()
            prefix = model.encode_image(image).cpu().numpy().astype(np.float32)

        img_ids_with_embeddings[img_id] = prefix

        if (i + 1) % 10000 == 0:
            with open(out_path, "wb") as f:
                pickle.dump(img_ids_with_embeddings, f)

    with open(out_path, "wb") as f:
        pickle.dump(img_ids_with_embeddings, f)

    print("Done")
    print("%0d embeddings saved " % len(img_ids_with_embeddings))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip_model_type",
        default="ViT-L/14@336px",
        # choices=("clip-vit-base-patch32"),
    )
    parser.add_argument(
        "--split", default="val2014", choices=("train2014", "val2014")
    )
    args = parser.parse_args()
    main(args.clip_model_type, args.split)
