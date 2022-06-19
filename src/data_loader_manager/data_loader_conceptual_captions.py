import os
import re
import sys
import time
import json
import copy
from tqdm import tqdm
import csv
import json
import torch
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import random
import cv2
import base64

from copy import deepcopy
from pprint import pprint
from easydict import EasyDict
from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import logging

logger = logging.getLogger(__name__)

from utils.dirs import create_dirs
from utils.vqa_tools import VQA
from utils.vqaEval import VQAEval
from utils.cache_system import save_cached_data, load_cached_data

from data_loader_manager.data_loader_wrapper import DataLoaderWrapper
from data_loader_manager.datasets import *

from torchvision.utils import make_grid, save_image

from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import urllib

import PIL.Image

from datasets import load_dataset


class DataLoaderConceptualCaptions(DataLoaderWrapper):
    """
    Data loader for VQA with ClipCap dataset
    """

    def __init__(self, config):
        DataLoaderWrapper.__init__(self, config)

    def LoadConceptualCaptions(self, module_config):
        """
        This function loads Conceptual Captions dataset
        """

        con_caps = load_dataset(
            "parquet",
            data_files={
                "train": module_config.config.conceptual_captions_path.train,
                "val": module_config.config.conceptual_captions_path.val,
            },
        )

        def str_to_list(example):
            example["caption"] = [example["caption"]]
            example["image_url"] = [example["image_url"]]
            return example

        con_caps = con_caps.map(str_to_list, batched=False)

        self.data.conceptual_captions = EasyDict(con_caps)

    def collate_fn(self, batch):

        image_urls = [sample["image_url"][0] for sample in batch]
        captions = [sample["caption"][0] for sample in batch]

        clip_embeddings = torch.stack(
            [torch.tensor(sample["clip_embeddings"]) for sample in batch]
        )

        tokenized_captions = self.tokenizer(
            captions,
            padding="longest",
            max_length=self.config.data_loader.additional.max_source_length,
            truncation=True,
            return_tensors="pt",
        )
        labels = tokenized_captions.input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels_attention_mask = tokenized_captions.attention_mask

        return {
            "image_urls": image_urls,
            "captions": captions,
            "clip_embeddings": clip_embeddings,
            "labels": labels,
            "labels_attention_mask": labels_attention_mask,
        }

    def set_dataloader(self):
        """
        This function wraps datasets into dataloader for trainers
        """

        self.train_dataset = self.data.conceptual_captions.train

        train_sampler = RandomSampler(self.train_dataset)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.config.train.batch_size,
            collate_fn=self.collate_fn,
            num_workers=8,
        )

        self.test_dataset = self.data.conceptual_captions.val
        test_sampler = SequentialSampler(self.test_dataset)
        self.test_dataloader = DataLoader(
            self.test_dataset,
            sampler=test_sampler,
            batch_size=self.config.valid.batch_size,
            collate_fn=self.collate_fn,
            num_workers=8,
        )
        logger.info(
            "[Data Statistics]: training data loader: {};  test data loader: {}".format(
                len(self.train_dataloader), len(self.test_dataloader)
            )
        )
