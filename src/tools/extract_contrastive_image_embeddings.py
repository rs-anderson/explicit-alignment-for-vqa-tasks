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

data_dir = Path("../../data")
vqa2_data_dir = data_dir / "vqa2"
okvqa_data_dir = data_dir / "ok-vqa"

def main(clip_model_name: str, split: str="val"):

    device = torch.device('cuda:0')
    
    # clip_model_name = clip_model_type.replace('/', '_')
    out_path = vqa2_data_dir / "pre-extracted_features" / f"coco_{clip_model_name}_{split}.pkl"
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    image_preprocessor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")

    with open(vqa2_data_dir / f'v2_OpenEnded_mscoco_{split}2014_questions.json', 'r') as f:
        data = json.load(f)
    
    # TODO: extract only "questions" from data dictionary
    print("%0d questions loaded from json " % len(data))

    all_embeddings = []
    all_captions = []
    
    for i in tqdm(range(len(data))):
        data_item = data[i]
        
        img_id = data_item["image_id"]
        img_path = okvqa_data_dir / f"{split}2014/COCO_{split}2014_{int(img_id):012d}.jpg"
        
        img_key_full = str(img_id).zfill(12)
        img = io.imread(img_path)

        image = image_preprocessor(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = model.get_image_features(image).cpu()
        
        data_item["clip_embedding"] = i
        
        all_embeddings.append(prefix)
        all_captions.append(data_item)
        
        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="clip-vit-base-patch32", choices=('clip-vit-base-patch32'))
    parser.add_argument('--split', default="val", choices=('train', 'val'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))