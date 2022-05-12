import math
import time
import os
import sys
import scipy
import datetime
import numpy as np
import json
import operator
import wandb
import logging
logger = logging.getLogger(__name__)

from pprint import pprint
from tqdm import tqdm
from easydict import EasyDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything

from .metrics_processors import MetricsProcessor
from utils.dirs import *

class BaseExecutor(pl.LightningModule, MetricsProcessor):
    additional_plugins = []
    
    def __init__(self, config, data_loader):
        super().__init__()
        self.config = config
        self.data_loader = data_loader

        logger.info(f'Initializing {self.__class__.__name__}...')
    
    def train_dataloader(self):
        return self.data_loader.train_dataloader
    
    def val_dataloader(self):
        # In many VQA dataset, the validation set is the test set
        return self.data_loader.test_dataloader
    
    def test_dataloader(self):
        return self.data_loader.test_dataloader

    def forward(self, **kwargs):
        return self.model(**kwargs)