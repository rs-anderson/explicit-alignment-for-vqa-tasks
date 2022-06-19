import torch
import argparse
from pathlib import Path
import clip
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from functools import partial

import PIL.Image

from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
import requests


USER_AGENT = get_datasets_user_agent()

num_threads = 20
data_dir = Path(
    "/home/rsa39/rds/rds-mlmi-2020-21-xyBFuSj0hm0/MLMI.2021-22/rsa39/project/RAVQA/data"
)
conceptual_captions_data_dir = data_dir / "conceptual_captions"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, image_preprocessor = clip.load("ViT-L/14@336px", device=device)


def fetch_single_image(image_url, timeout=10, retries=0):
    for _ in range(retries + 1):
        try:
            response = requests.get(image_url, stream=True, timeout=timeout)
            if response:
                image = PIL.Image.open(response.raw)
                break
            else:
                image = None
        except Exception:
            image = None
    return image


def get_embeddings_from_images(batch, num_threads, timeout=10, retries=0):
    fetch_single_image_with_args = partial(
        fetch_single_image, timeout=timeout, retries=retries
    )
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch_images = list(
            executor.map(fetch_single_image_with_args, batch["image_url"])
        )

    batch_images_preprocessed = []
    batch_image_urls = []
    batch_captions = []

    for img, image_url, caption in zip(
        batch_images, batch["image_url"], batch["caption"]
    ):
        if img is not None:
            batch_images_preprocessed.append(image_preprocessor(img))
            batch_image_urls.append(image_url)
            batch_captions.append(add_period(caption))

    batch_images_preprocessed = torch.stack(batch_images_preprocessed).to(
        device
    )

    clip_embeddings = encode_images(batch_images_preprocessed)

    batch["clip_embeddings"] = clip_embeddings
    batch["image_url"] = batch_image_urls
    batch["caption"] = batch_captions

    print(f"{clip_embeddings.shape[0]} valid images in batch")
    return batch


def encode_images(images):
    with torch.no_grad():
        clip_embeddings = (
            model.encode_image(images).cpu().numpy().astype(np.float32)
        )
    return clip_embeddings


def add_period(caption: str):
    caption = caption.strip()
    if caption[-1] != ".":
        caption = caption + "."
    elif caption[-2] == " ":
        caption = caption[:-2] + "."
    return caption


def main(clip_model_name: str, subtype: str = "validation"):

    print(f"Extracting {subtype} using {clip_model_name}")

    clip_model_name = clip_model_name.replace("/", "_")
    out_path = (
        conceptual_captions_data_dir
        / "pre-extracted-features"
        / f"conceptual_captions_{clip_model_name}_{subtype}.parquet"
    )

    dataset_without_embeddings = load_dataset(
        "conceptual_captions", split=f"{subtype}"
    )

    dataset_with_embeddings = dataset_without_embeddings.map(
        get_embeddings_from_images,
        batched=True,
        batch_size=512,
        fn_kwargs={
            "num_threads": num_threads,
        },
    )

    print(f"Writing output to {out_path}...")
    dataset_with_embeddings.to_parquet(out_path)
    print(f"Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip_model_type",
        default="ViT-L/14@336px",
        choices=("ViT-L/14@336px", "ViT-B/32"),
    )
    parser.add_argument("--split", default="train[:2]")
    args = parser.parse_args()
    main(args.clip_model_type, args.split)
