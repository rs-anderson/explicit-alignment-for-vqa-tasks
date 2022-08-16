import math
from re import I
import time
import os
import sys
from typing import Optional
import scipy
import datetime
import numpy as np
import json
import operator

# from models.clip_predict import generate2, generate_beam
from trainers.base_executor import BaseExecutor
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
from .base_executor import BaseExecutor
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from utils.dirs import *
from models.vct0 import VCT0Prefix



# from models.clip_predict import *


class FewShotVQAExecutor(BaseExecutor):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)

        if config.mode == "train":
            self.train_data_loader = self.data_loader.train_dataloader
        else:
            self.train_data_loader = None
        self.test_data_loader = self.data_loader.test_dataloader

        self.tokenizer = data_loader.tokenizer
        self.decoder_tokenizer = data_loader.decoder_tokenizer

        ModelClass = globals()[self.config.model_config.ModelClass]
        self.model = ModelClass(**self.config.model_config.model_args)

        self.tokenizer.bos_token = self.tokenizer.pad_token
        # self.model.gpt.resize_token_embeddings(len(self.tokenizer))

    def configure_optimizers(self):
        """
        Return optimizers and schedulers
        """

        optimization_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()],
                "lr": self.config.train.lr,
                "initial_lr": self.config.train.lr,
            },
        ]

        for group in optimization_parameters:
            logger.info(
                "#params: {}   lr: {}".format(
                    len(group["params"]), group["lr"]
                )
            )

        """define optimizer"""
        self.optimizer = torch.optim.AdamW(
            optimization_parameters, lr=self.config.train.lr
        )

        if self.config.train.scheduler == "linear":
            from transformers import get_linear_schedule_with_warmup

            # Using Linear scheduler
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.train.additional.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
                last_epoch=self.global_step,
            )
        elif self.config.train.scheduler == "cosine":
            t_total = self.config.train.epochs
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                t_total,
                eta_min=1e-5,
                last_epoch=-1,
                verbose=False,
            )
        else:
            from transformers import get_constant_schedule_with_warmup

            # Using constant scheduler
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.train.additional.warmup_steps,
                last_epoch=self.global_step,
            )

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                # REQUIRED: The scheduler instance
                "scheduler": self.scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "step",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": None,
            },
        }

    def training_step(self, sample_batched, batch_idx):
        return None

    def validation_step(self, sample_batched, batch_idx):
        return self._generative_step(sample_batched, batch_idx)

    def validation_epoch_end(self, validation_step_outputs):
        log_dict = self.evaluate_outputs(validation_step_outputs)
        self.logging_results(log_dict)
        return log_dict.metrics

    def test_step(self, sample_batched, batch_idx):
        return self._generative_step(sample_batched, batch_idx)

    def test_epoch_end(self, validation_step_outputs):
        log_dict = self.evaluate_outputs(validation_step_outputs)
        self.logging_results(log_dict, prefix=self.config.test.evaluation_name)
        return log_dict.metrics

    def _generative_step(self, sample_batched, batch_idx):
        """
        This function is shared by both valid and test
        """
        predictions = []
        table_entries = []

        labels = sample_batched["labels"]
        test_batch = EasyDict(
            {
                "input_ids": sample_batched["generative_input_ids"].to(
                    self.device
                ),
                "attention_mask": sample_batched[
                    "generative_attention_mask"
                ].to(self.device),
                "clip_embeddings": sample_batched["clip_embeddings"].to(
                    self.device
                ),
                "max_length": self.config.data_loader.additional.max_target_length,
            }
        )

        if "decoder_generative_input_ids" in sample_batched:
            test_batch["decoder_generative_input_ids"] = sample_batched["decoder_generative_input_ids"][:, :-1].to(self.device)
            test_batch["decoder_generative_attention_mask"] = sample_batched["decoder_generative_attention_mask"][:, :-1].to(self.device)


        if self.config.data_loader.additional.pass_examples_through_encoder_one_at_a_time:
            test_batch.input_ids = test_batch.input_ids.view(-1, self.config.data_loader.additional.num_shots+1, test_batch.input_ids.shape[-1])
            test_batch.attention_mask = test_batch.attention_mask.view(-1, self.config.data_loader.additional.num_shots+1, test_batch.attention_mask.shape[-1])
        
        if self.config.data_loader.additional.ensemble_one_shots:
            test_batch.input_ids = test_batch.input_ids.view(-1, self.config.data_loader.additional.num_shots, test_batch.input_ids.shape[-1])
            test_batch.attention_mask = test_batch.attention_mask.view(-1, self.config.data_loader.additional.num_shots, test_batch.attention_mask.shape[-1])
            outputs = self.generate_from_ensembles(test_batch, num_ensembles=self.config.data_loader.additional.num_shots, num_shots=1)

        elif self.config.data_loader.additional.num_permutations_of_in_context_examples > 0:
            test_batch.input_ids = test_batch.input_ids.view(-1, self.config.data_loader.additional.num_permutations_of_in_context_examples, test_batch.input_ids.shape[-1])
            test_batch.attention_mask = test_batch.attention_mask.view(-1, self.config.data_loader.additional.num_permutations_of_in_context_examples, test_batch.input_ids.shape[-1])
            outputs = self.generate_from_ensembles(test_batch, num_ensembles=self.config.data_loader.additional.num_permutations_of_in_context_examples)

        else:
            outputs = self.model.generate(
                question_tokens=test_batch.input_ids,
                question_mask=test_batch.attention_mask,
                prefix=test_batch.clip_embeddings,
                decoder_input_ids=test_batch.get("decoder_generative_input_ids", None),
                decoder_attention_mask=test_batch.get("decoder_generative_attention_mask", None),
                no_prefix=self.config.data_loader.additional.no_prefix,
                pass_examples_through_encoder_one_at_a_time=self.config.data_loader.additional.pass_examples_through_encoder_one_at_a_time,
                max_length=test_batch.max_length,
            )

        bos_token_id = self.decoder_tokenizer.bos_token_id
        for index, i in enumerate(labels):

            cleaned_i = [
                label if label != -100 else self.decoder_tokenizer.pad_token_id
                for label in i
            ]
            cleaned_i = torch.LongTensor(cleaned_i)
            decoded_label = self.decoder_tokenizer.decode(
                cleaned_i, skip_special_tokens=True
            )
            # print(self.tokenizer.decode(cleaned_i, skip_special_tokens=True))
            if self.config.data_loader.additional.pass_examples_through_encoder_one_at_a_time and self.config.data_loader.additional.no_prefix:
                output_sequence = outputs[index][0].cpu().numpy().astype(int).tolist()
            else:
                output_sequence = outputs[index].cpu().numpy().astype(int).tolist()
            # print('output_sequence', output_sequence)

            # if bos_token_id in output_sequence:
            #     output_sequence = output_sequence[
            #         output_sequence.index(bos_token_id) :
            #     ]

            # print('output_sequence after', output_sequence)
            decoded_output = self.decoder_tokenizer.decode(
                output_sequence, skip_special_tokens=True
            )
            actual_output = self.decoder_tokenizer.decode(
                output_sequence, skip_special_tokens=False
            )
            # print(self.tokenizer.decode(cleaned_i, skip_special_tokens=True))

            if batch_idx < 1:
                print(
                    decoded_label,
                    "<--->",
                    decoded_output,
                    "   ({})".format(actual_output),
                )

            question_id = sample_batched["question_ids"][index]
            predictions.append(
                {
                    "question_id": question_id,
                    "answer": decoded_output,
                }
            )

            item = self.data_loader.data.vqa_data.lookup[str(question_id)]
            
            if self.config.data_loader.additional.pass_examples_through_encoder_one_at_a_time or self.config.data_loader.additional.ensemble_one_shots:
                input_to_decode = [token for input_list in test_batch.input_ids[index].cpu().tolist() for token in input_list]
            
            elif self.config.data_loader.additional.num_permutations_of_in_context_examples > 0:
                input_to_decode = test_batch.input_ids[index][0]

            else:
                input_to_decode = sample_batched["generative_input_ids"][index]

            table_entry = [
                question_id,
                item["img_key"],
                # sample_batched["in_context_img_keys"][index],
                item["question"],
                self.tokenizer.decode(input_to_decode),
                item["answers"],
                item["gold_answer"],
                decoded_output,
            ]
            table_entries.append(table_entry)

        data_to_return = {
            "predictions": predictions,
            "outputs": outputs,
            "question_ids": sample_batched["question_ids"],
            "answers": sample_batched["answers"],
            "table_entries": table_entries,
        }

        return data_to_return

    def generate_from_ensembles(self, test_batch: EasyDict, num_ensembles: int, num_shots: Optional[int] = None):
        ensembled_outputs = []
        batch_sequence_scores = np.zeros((test_batch.input_ids.shape[0], num_ensembles))
        for i in range(num_ensembles):
            
            if self.config.data_loader.additional.ensemble_one_shots:
                clip_embeddings = test_batch.clip_embeddings[:, [i, -1]]

            elif self.config.data_loader.additional.num_permutations_of_in_context_examples > 0:
                clip_embeddings = test_batch.clip_embeddings[:, i]

            outputs = self.model.generate(
                question_tokens=test_batch.input_ids[:, i],
                question_mask=test_batch.attention_mask[:, i],
                prefix=clip_embeddings,
                no_prefix=self.config.data_loader.additional.no_prefix,
                pass_examples_through_encoder_one_at_a_time=self.config.data_loader.additional.pass_examples_through_encoder_one_at_a_time,
                num_shots=num_shots,
                max_length=test_batch.max_length,
                output_scores=True,
                return_dict_in_generate=True,
            )

            outputs_scores = torch.log(torch.stack(outputs.scores).softmax(dim=-1))

            for j, sequence in enumerate(outputs.sequences):
                sequence_score = 0
                for k, input_id in enumerate(sequence):
                    if input_id not in [0, 1, 2]:
                        score = outputs_scores[k-1, j, input_id]
                        sequence_score += score 
                batch_sequence_scores[j, i] = sequence_score
            
            # num_permutations_of_in_context_examplesd_scores.append(sequence_score)
            ensembled_outputs.append(outputs.sequences)

        best_output_from_ensembles_ind = np.argmax(batch_sequence_scores, axis=1)
        best_output_from_ensembles = [ensembled_outputs[ind][i] for  i, ind in enumerate(best_output_from_ensembles_ind)]

        return best_output_from_ensembles

    def evaluate_outputs(self, step_outputs, mode="test"):
        # Batching every validation step outputs
        batch_predictions = []

        columns = [
            "question_id",
            "image_key",
            "question",
            "input",
            "answers",
            "gold_answer",
            "prediction",
        ]
        test_table = wandb.Table(columns=columns)

        for i, step_output in enumerate(step_outputs):
            batch_predictions += step_output["predictions"]

            if i < 10:
                for table_entry in step_output["table_entries"]:
                    test_table.add_data(*table_entry)

        ##############################
        ##    Compute Metrics       ##
        ##############################
        data_used_for_metrics = EasyDict(
            mode=mode,
            epoch=self.current_epoch,
            batch_predictions=batch_predictions,
        )

        log_dict = self.compute_metrics(data_used_for_metrics)
        log_dict.artifacts.test_table = test_table

        return log_dict

    def logging_results(self, log_dict, prefix="test"):

        ### Add test results to wandb / tensorboard
        metrics_to_log = EasyDict()
        wandb_artifacts_to_log = dict()
        # Refractor the column names
        for metric, value in log_dict.metrics.items():
            metrics_to_log[f"{prefix}/{metric}"] = value

        # include other artifacts / metadata
        metrics_to_log[f"{prefix}/epoch"] = self.current_epoch
        wandb_artifacts_to_log.update(
            {
                f"retrieval_predictions_epoch{self.current_epoch}_MODE({self.config.mode})_SET(TEST)": log_dict.artifacts[
                    "test_table"
                ]
            }
        )
        pprint(metrics_to_log)
        pprint(wandb_artifacts_to_log)

        logger.info(
            f"Evaluation results [{self.trainer.state.stage}]: {metrics_to_log}"
        )

        if self.trainer.state.stage in ["sanity_check"]:
            logging.warning("Sanity check mode, not saving to loggers.")
            return

        # Add to loggers
        for metric, value in metrics_to_log.items():
            if type(value) in [float, int, np.float64]:
                self.log(metric, float(value), logger=True)
            else:
                logger.info(
                    f"{metric} is not a type that can be logged, skippped."
                )

        # Call wandb to log artifacts; remember to use commit=False so that the data will be logged
        #       with other metrics later.
        if self.config.args.log_prediction_tables:
            self.wandb_logger.experiment.log(
                wandb_artifacts_to_log, commit=False
            )

    def forward(self, **kwargs):
        return self.model(**kwargs)
