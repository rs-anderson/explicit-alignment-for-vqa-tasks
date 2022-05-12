"""Finetuning script for RAG models. Adapted from examples.seq2seq.finetune.py"""

import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed as torch_distrib
from pytorch_lightning.plugins.training_type import DDPPlugin
from torch.utils.data import DataLoader

from transformers import (
    AutoConfig,
    AutoTokenizer,
    BartForConditionalGeneration,
    BatchEncoding,
    RagConfig,
    # RagTokenForGeneration,
    # RagSequenceForGeneration,
    RagTokenizer,
    T5ForConditionalGeneration,
)
from rag_model import RagSequenceForGeneration, RagTokenForGeneration
from transformers import logging as transformers_logging
from transformers.integrations import is_ray_available

from vqa_tools import VQA
from vqaEval import VQAEval

if is_ray_available():
    import ray
    from distributed_ray_retriever import RagRayDistributedRetriever, RayRetriever

from callbacks_rag import (  # noqa: E402 # isort:skipq
    get_checkpoint_callback,
    get_early_stopping_callback,
    Seq2SeqLoggingCallback,
)

from distributed_pytorch_retriever import RagPyTorchDistributedRetriever  # noqa: E402 # isort:skip
from utils_rag import (  # noqa: E402 # isort:skip
    calculate_exact_match,
    flatten_list,
    get_git_info,
    is_rag_model,
    lmap,
    pickle_save,
    save_git_info,
    save_json,
    set_extra_model_params,
    Seq2SeqDataset,
)

# need the parent dir module
sys.path.insert(2, str(Path(__file__).resolve().parents[1]))
from lightning_base import BaseTransformer, add_generic_args, generic_train  # noqa


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

transformers_logging.set_verbosity_info()


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class CustomDDP(DDPPlugin):
    def init_ddp_connection(self, global_rank=None, world_size=None) -> None:
        module = self.model
        global_rank = global_rank if global_rank is not None else self.cluster_environment.global_rank()
        world_size = world_size if world_size is not None else self.cluster_environment.world_size()
        os.environ["MASTER_ADDR"] = self.cluster_environment.master_address()
        os.environ["MASTER_PORT"] = str(self.cluster_environment.master_port())
        if not torch.distributed.is_initialized():
            logger.info(f"initializing ddp: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}")
            torch_distrib.init_process_group(self.torch_distributed_backend, rank=global_rank, world_size=world_size)

        if module.is_rag_model:
            self.distributed_port = module.hparams.distributed_port
            if module.distributed_retriever == "pytorch":
                module.model.rag.retriever.init_retrieval(self.distributed_port)
            elif module.distributed_retriever == "ray" and global_rank == 0:
                # For the Ray retriever, only initialize it once when global
                # rank is 0.
                module.model.rag.retriever.init_retrieval()


class GenerativeQAModule(BaseTransformer):
    mode = "generative_qa"
    loss_names = ["loss"]
    metric_names = ["em"]
    val_metric = "em"

    def __init__(self, hparams, **kwargs):
        # when loading from a pytorch lightning checkpoint, hparams are passed as dict
        if isinstance(hparams, dict):
            hparams = AttrDict(hparams)
        if hparams.model_type == "rag_sequence":
            self.model_class = RagSequenceForGeneration
        elif hparams.model_type == "rag_token":
            self.model_class = RagTokenForGeneration
        elif hparams.model_type == "bart":
            self.model_class = BartForConditionalGeneration
        else:
            self.model_class = T5ForConditionalGeneration
        self.is_rag_model = is_rag_model(hparams.model_type)

        config_class = RagConfig if self.is_rag_model else AutoConfig
        config = config_class.from_pretrained(hparams.model_name_or_path)

        # set retriever parameters
        config.index_name = hparams.index_name or config.index_name
        config.passages_path = hparams.passages_path or config.passages_path
        config.index_path = hparams.index_path or config.index_path
        config.use_dummy_dataset = hparams.use_dummy_dataset

        # set extra_model_params for generator configs and load_model
        extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "attention_dropout", "dropout")
        if self.is_rag_model:
            if hparams.prefix is not None:
                config.generator.prefix = hparams.prefix
            config.label_smoothing = hparams.label_smoothing
            hparams, config.generator = set_extra_model_params(extra_model_params, hparams, config.generator)
            if hparams.distributed_retriever == "pytorch":
                retriever = RagPyTorchDistributedRetriever.from_pretrained(hparams.model_name_or_path, config=config)
            elif hparams.distributed_retriever == "ray":
                # The Ray retriever needs the handles to the retriever actors.
                retriever = RagRayDistributedRetriever.from_pretrained(
                    hparams.model_name_or_path, hparams.actor_handles, config=config
                )
            model = self.model_class.from_pretrained(hparams.model_name_or_path, config=config, retriever=retriever)
            prefix = config.question_encoder.prefix
        else:
            if hparams.prefix is not None:
                config.prefix = hparams.prefix
            hparams, config = set_extra_model_params(extra_model_params, hparams, config)
            model = self.model_class.from_pretrained(hparams.model_name_or_path, config=config)
            prefix = config.prefix

        tokenizer = (
            RagTokenizer.from_pretrained(hparams.model_name_or_path)
            if self.is_rag_model
            else AutoTokenizer.from_pretrained(hparams.model_name_or_path)
        )

        super().__init__(hparams, config=config, tokenizer=tokenizer, model=model)

        # print(self.model)
        print('resizing the model embeddings to accommodate special tokens!')
        self.model.generator.resize_token_embeddings(len(self.tokenizer.generator))
        self.model.question_encoder.resize_token_embeddings(len(self.tokenizer.question_encoder))
        

        save_git_info(self.hparams.output_dir)
        self.output_dir = Path(self.hparams.output_dir)
        if hparams.do_train:
            self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        else:
            self.metrics_save_path = Path(self.output_dir) / "metrics_test.json"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        pickle_save(self.hparams, self.hparams_save_path)
        self.step_count = 0
        self.metrics = defaultdict(list)

        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            prefix=prefix or "",
        )
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_max_target_length,
            "test": self.hparams.test_max_target_length,
        }
        assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
        assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"

        self.hparams.git_sha = get_git_info()["repo_sha"]
        self.num_workers = hparams.num_workers
        self.distributed_port = self.hparams.distributed_port

        # For single GPU training, init_ddp_connection is not called.
        # So we need to initialize the retrievers here.
        if hparams.gpus <= 1:
            if hparams.distributed_retriever == "ray":
                self.model.retriever.init_retrieval()
            elif hparams.distributed_retriever == "pytorch":
                self.model.retriever.init_retrieval(self.distributed_port)

        self.distributed_retriever = hparams.distributed_retriever


        # Load VQA helpers
        self.vqa_helpers = {
            'train': VQA('/home/wl356/projects/Knowledge-based-visual-question-answering/data/ok-vqa/mscoco_train2014_annotations.json', 
                            '/home/wl356/projects/Knowledge-based-visual-question-answering/data/ok-vqa/OpenEnded_mscoco_train2014_questions.json'),
            'test': VQA('/home/wl356/projects/Knowledge-based-visual-question-answering/data/ok-vqa/mscoco_val2014_annotations.json', 
                            '/home/wl356/projects/Knowledge-based-visual-question-answering/data/ok-vqa/OpenEnded_mscoco_val2014_questions.json'),
        }
        # input('init done.')


    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def _step(self, batch: dict, is_training: bool = False) -> Tuple:
        source_ids, source_mask, target_ids = batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"]
        
        rag_kwargs = {}
        if isinstance(self.model, T5ForConditionalGeneration):
            decoder_input_ids = self.model._shift_right(target_ids)
            lm_labels = target_ids
        elif isinstance(self.model, BartForConditionalGeneration):
            decoder_input_ids = target_ids[:, :-1].contiguous()
            lm_labels = target_ids[:, 1:].clone()
        else:
            assert self.is_rag_model
            generator = self.model.rag.generator
            if isinstance(generator, T5ForConditionalGeneration):
                decoder_start_token_id = generator.config.decoder_start_token_id
                decoder_input_ids = (
                    torch.cat(
                        [torch.tensor([[decoder_start_token_id]] * target_ids.shape[0]).to(target_ids), target_ids],
                        dim=1,
                    )
                    if target_ids.shape[0] < self.target_lens["train"]
                    else generator._shift_right(target_ids)
                )
            elif isinstance(generator, BartForConditionalGeneration):
                decoder_input_ids = target_ids
            lm_labels = decoder_input_ids
            # print('target_ids', target_ids)
            # print(self.tokenizer.generator.batch_decode(target_ids))
            # print('decoder_input_ids', decoder_input_ids)
            # print(self.tokenizer.generator.batch_decode(decoder_input_ids))
            rag_kwargs["reduce_loss"] = True

        assert decoder_input_ids is not None

        outputs = self(
            source_ids,
            attention_mask=source_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            output_retrieved=True,
            labels=lm_labels,
            **rag_kwargs,
        )
        # print(outputs)
        loss = outputs["loss"]
        retrieved_doc_ids = outputs['retrieved_doc_ids']

        # if is_training:
        #     # print(loss)
        #     doc_scores = outputs['doc_scores']
        #     batch_answers = batch['answers']
        #     # print(batch_answers)
        #     # print(batch['passage_pos_items'])
        #     # doc_scores = F.softmax(doc_scores, dim=-1)
        #     doc_scores = F.sigmoid(doc_scores)

        #     labels = self.cal_retrieval_metrics(batch_answers, 
        #                             retrieved_doc_ids, 
        #                             return_label=True).to(loss.device)

        #     retrieval_loss = F.binary_cross_entropy(doc_scores, labels)
        #     loss += 0.001*retrieval_loss

        
        return (loss,), retrieved_doc_ids



    def cal_retrieval_metrics(self, batch_answers, retrieved_doc_ids, return_label=False):
        def most_frequent(List):
            return max(set(List), key = List.count)
        retrieved_doc_ids = np.array(retrieved_doc_ids)
        retrieved_docs = self.model.retriever.index.get_doc_dicts(retrieved_doc_ids)
        log_result = {
            'recall': [],
            'precision': [],
            'gold_precision': [],
            'gold_recall': [],
        }
        labels = []
        for answer_list, docs in zip(batch_answers, retrieved_docs):
            
            filtered_answer_list = [ans for ans in answer_list if ans != '']
            gold_answer = most_frequent(filtered_answer_list)

            unique_answers = list(set(answer_list))
            doc_texts = docs['text']
            
            found_answers = []
            found_gold_answers = []
            K = len(doc_texts)
            this_batch_labels = [0] * len(doc_texts)

            for index, passage_data in enumerate(doc_texts):
                for answer in unique_answers:
                    if answer.lower() in passage_data.lower():
                        found_answers.append(answer)
                        this_batch_labels[index] = 1
                        break
                if gold_answer.lower() in passage_data.lower():
                    found_gold_answers.append(answer)
                    this_batch_labels[index] = 1

            labels.append(this_batch_labels)
                    
            if len(found_answers) > 0:
                # At least one answer is retireved
                log_result['recall'].append(1)
            else:
                log_result['recall'].append(0)
            # The proportion of retrieved knowledge has an answer
            log_result['precision'].append(len(found_answers) / K)

            if len(found_gold_answers) > 0:
                # if gold answer is found
                log_result['gold_recall'].append(1)
            else:
                log_result['gold_recall'].append(0)
            # The proportion of retrieved knowledge has the gold answer
            log_result['gold_precision'].append(len(found_gold_answers) / K)

        if not return_label:
            for metric in log_result.keys():
                log_result[metric] = np.mean(np.array(log_result[metric]))
            
            return log_result
        else:
            return torch.Tensor(labels)

    @property
    def pad(self) -> int:
        raise NotImplementedError("pad not implemented")

    def training_step(self, batch, batch_idx) -> Dict:
        loss_tensors, retrieved_doc_ids = self._step(batch, is_training=True)
        
        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        # tokens per batch
        tgt_pad_token_id = (
            self.tokenizer.generator.pad_token_id
            if isinstance(self.tokenizer, RagTokenizer)
            else self.tokenizer.pad_token_id
        )
        src_pad_token_id = (
            self.tokenizer.question_encoder.pad_token_id
            if isinstance(self.tokenizer, RagTokenizer)
            else self.tokenizer.pad_token_id
        )
        logs["tpb"] = (
            batch["input_ids"].ne(src_pad_token_id).sum() + batch["decoder_input_ids"].ne(tgt_pad_token_id).sum()
        )
        logs.update({
                "retrieved_doc_ids": retrieved_doc_ids,
                "answers": batch['answers']
        })

        return {"loss": loss_tensors[0], 
                "log": logs}

    def training_epoch_end(self, outs):
        ############# Check retrieval metrics #############
        all_answers = []
        retrieved_doc_ids = []
        for x in outs:
            all_answers += x['log']['answers']
            retrieved_doc_ids += x['log']['retrieved_doc_ids'].numpy().tolist()

        retrieval_results = self.cal_retrieval_metrics(all_answers, retrieved_doc_ids)
        retrieval_results['step_count'] = self.step_count
        # print(retrieval_results)
        self.save_metrics(retrieval_results, 'train')

        print('End of epoch, saving models...')
        save_path = self.output_dir.joinpath("checkpoint_{}".format(self.step_count))
        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def validation_step(self, batch, batch_idx) -> Dict:
        if self.step_count in [0, 1, 2, 4, 5, 6]:
            print('skip validation!')
            return {}
        return self._generative_step(batch)

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.step_count += 1
        
        if outputs[0] == {}:
            return None
        
        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        loss = losses["loss"]
        gen_metrics = {
            k: np.array([x[k] for x in outputs]).mean() for k in self.metric_names + ["gen_time", "gen_len"]
        }
        metrics_tensor: torch.FloatTensor = torch.tensor(gen_metrics[self.val_metric]).type_as(loss)
        gen_metrics.update({k: v.item() for k, v in losses.items()})

        '''
        answers.append({
                    'question_id': question_id,
                    'answer': decoded_output,
                })
        '''
        answers = []
        for x in outputs:
            for question_id, decoded_output in zip(x['question_ids'], x['preds']):
                answers.append({
                    'question_id': question_id,
                    'answer': decoded_output,
                })
        

        # fix for https://github.com/PyTorchLightning/pytorch-lightning/issues/2424
        if dist.is_initialized():
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            metrics_tensor = metrics_tensor / dist.get_world_size()
            gen_metrics.update({self.val_metric: metrics_tensor.item()})

        losses.update(gen_metrics)
        metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        metrics["step_count"] = self.step_count

        try:
            # create vqa object and vqaRes object
            vqa_helper = self.vqa_helpers['test']
            vqaRes = vqa_helper.loadResFromDict(answers)

            # create vqaEval object by taking vqa and vqaRes
            vqaEval = VQAEval(vqa_helper, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2

            # evaluate results
            """
            If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
            By default it uses all the question ids in annotation file
            """
            vqaEval.evaluate()

            # print accuracies
            print ("Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall']))
            print ("Per Question Type Accuracy is the following:")
            for quesType in vqaEval.accuracy['perQuestionType']:
                print ("%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType]))
            print ("\n")
            print ("Per Answer Type Accuracy is the following:")
            for ansType in vqaEval.accuracy['perAnswerType']:
                print ("%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType]))
            print ("\n")
            metrics['overall'] = vqaEval.accuracy['overall']
            for quesType in vqaEval.accuracy['perQuestionType']:
                metrics['quesType_{}'.format(quesType)] = vqaEval.accuracy['perQuestionType'][quesType]
            for ansType in vqaEval.accuracy['perAnswerType']:
                metrics['ansType_{}'.format(ansType)] = vqaEval.accuracy['perAnswerType'][ansType]
            
        except Exception as e:
            print(e)
            print('May not validate on full set. Skip!')


        ############# Check retrieval metrics #############
        all_answers = []
        retrieved_doc_ids = []
        for x in outputs:
            all_answers += x['answers']
            retrieved_doc_ids += x['retrieved_doc_ids'].numpy().tolist()

        retrieval_results = self.cal_retrieval_metrics(all_answers, retrieved_doc_ids)
        metrics.update(retrieval_results)


        self.save_metrics(metrics, prefix)  # writes to self.metrics_save_path
        preds = flatten_list([x["preds"] for x in outputs])
        return {"log": metrics, "preds": preds, f"{prefix}_loss": loss, f"{prefix}_{self.val_metric}": metrics_tensor}

    def save_metrics(self, latest_metrics, type_path) -> None:
        self.metrics[type_path].append(latest_metrics)
        save_json(self.metrics, self.metrics_save_path)

    def calc_generative_metrics(self, preds, target) -> Dict:
        return calculate_exact_match(preds, target)

    def _generative_step(self, batch: dict) -> dict:
        question_ids = batch.pop('question_ids')
        answers = batch.pop('answers')
        passage_pos_items = batch.pop('passage_pos_items')
        start_time = time.time()
        batch = BatchEncoding(batch).to(device=self.model.device)
        # print(self.tokenizer.question_encoder.batch_decode(batch["input_ids"]))
        generation_results = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            do_deduplication=False,  # rag specific parameter
            use_cache=True,
            min_length=1,
            max_length=self.target_lens["val"],
        )
        generated_ids = generation_results['generation_output']
        doc_ids = generation_results['retrieved_doc_ids']
        # print(batch)
        gen_time = (time.time() - start_time) / batch["input_ids"].shape[0]
        preds: List[str] = self.ids_to_clean_text(generated_ids)
        target: List[str] = self.ids_to_clean_text(batch["decoder_input_ids"])
        loss_tensors, _ = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        gen_metrics: Dict = self.calc_generative_metrics(preds, target)

        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(gen_time=gen_time, gen_len=summ_len, preds=preds, target=target, **gen_metrics)
        
        # Put non-encodable items back
        base_metrics.update(question_ids=question_ids, answers=answers, retrieved_doc_ids=doc_ids, passage_pos_items=passage_pos_items)
        
        # print('base_metrics', base_metrics)

        return base_metrics

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")

    def get_dataset(self, type_path) -> Seq2SeqDataset:
        n_obs = self.n_obs[type_path]
        max_target_length = self.target_lens[type_path]
        dataset = Seq2SeqDataset(
            self.tokenizer,
            type_path=type_path,
            n_obs=n_obs,
            max_target_length=max_target_length,
            **self.dataset_kwargs,
        )
        return dataset

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self.get_dataset(type_path)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)

    # @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        print('saving checkpoints!')
        save_path = self.output_dir.joinpath("checkpoint{}".format(self.step_count))
        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        add_generic_args(parser, root_dir)
        parser.add_argument(
            "--max_source_length",
            default=128,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=25,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--val_max_target_length",
            default=25,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--test_max_target_length",
            default=25,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument("--logger_name", type=str, choices=["default", "wandb", "wandb_shared"], default="default")
        parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_val", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument(
            "--prefix",
            type=str,
            default=None,
            help="Prefix added at the beginning of each text, typically used with T5-based models.",
        )
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            required=False,
            help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
        )
        parser.add_argument(
            "--distributed-port", type=int, default=-1, required=False, help="Port number for distributed training."
        )
        parser.add_argument(
            "--model_type",
            choices=["rag_sequence", "rag_token", "bart", "t5"],
            type=str,
            help="RAG model type: sequence or token, if none specified, the type is inferred from the model_name_or_path",
        )
        return parser

    @staticmethod
    def add_retriever_specific_args(parser):
        parser.add_argument(
            "--index_name",
            type=str,
            default=None,
            help="Name of the index to use: 'hf' for a canonical dataset from the datasets library (default), 'custom' for a local index, or 'legacy' for the orignal one)",
        )
        parser.add_argument(
            "--passages_path",
            type=str,
            default=None,
            help="Path to the dataset of passages for custom index. More info about custom indexes in the RagRetriever documentation as well as in `examples/rag/use_own_knowledge_dataset.py`",
        )
        parser.add_argument(
            "--index_path",
            type=str,
            default=None,
            help="Path to the faiss index for custom index. More info about custom indexes in the RagRetriever documentation as well as in `examples/rag/use_own_knowledge_dataset.py`",
        )
        parser.add_argument(
            "--distributed_retriever",
            choices=["ray", "pytorch"],
            type=str,
            default="pytorch",
            help="What implementation to use for distributed retriever? If "
            "pytorch is selected, the index is loaded on training "
            "worker 0, and torch.distributed is used to handle "
            "communication between training worker 0, and the other "
            "training workers. If ray is selected, the Ray library is "
            "used to create load the index on separate processes, "
            "and Ray handles the communication between the training "
            "workers and the retrieval actors.",
        )
        parser.add_argument(
            "--use_dummy_dataset",
            type=bool,
            default=False,
            help="Whether to use the dummy version of the dataset index. More info about custom indexes in the RagRetriever documentation as well as in `examples/rag/use_own_knowledge_dataset.py`",
        )
        return parser

    @staticmethod
    def add_ray_specific_args(parser):
        # Ray cluster address.
        parser.add_argument(
            "--ray-address",
            default="auto",
            type=str,
            help="The address of the Ray cluster to connect to. If not "
            "specified, Ray will attempt to automatically detect the "
            "cluster. Has no effect if pytorch is used as the distributed "
            "retriever.",
        )
        parser.add_argument(
            "--num_retrieval_workers",
            type=int,
            default=1,
            help="The number of retrieval actors to use when Ray is selected"
            "for the distributed retriever. Has no effect when "
            "distributed_retriever is set to pytorch.",
        )
        return parser


def main(args=None, model=None) -> GenerativeQAModule:
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = GenerativeQAModule.add_model_specific_args(parser, os.getcwd())
    parser = GenerativeQAModule.add_retriever_specific_args(parser)

    args = args or parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)

    named_actors = []
    if args.distributed_retriever == "ray" and args.gpus > 1:
        if not is_ray_available():
            raise RuntimeError("Please install Ray to use the Ray " "distributed retriever.")
        # Connect to an existing Ray cluster.
        try:
            ray.init(address=args.ray_address)
        except (ConnectionError, ValueError):
            logger.warning(
                "Connection to Ray cluster failed. Make sure a Ray"
                "cluster is running by either using Ray's cluster "
                "launcher (`ray up`) or by manually starting Ray on "
                "each node via `ray start --head` for the head node "
                "and `ray start --address='<ip address>:6379'` for "
                "additional nodes. See "
                "https://docs.ray.io/en/master/cluster/index.html "
                "for more info."
            )
            raise

        # Create Ray actors only for rank 0.
        if ("LOCAL_RANK" not in os.environ or int(os.environ["LOCAL_RANK"]) == 0) and (
            "NODE_RANK" not in os.environ or int(os.environ["NODE_RANK"]) == 0
        ):
            remote_cls = ray.remote(RayRetriever)
            named_actors = [
                remote_cls.options(name="retrieval_worker_{}".format(i)).remote()
                for i in range(args.num_retrieval_workers)
            ]
        else:
            logger.info(
                "Getting named actors for NODE_RANK {}, LOCAL_RANK {}".format(
                    os.environ["NODE_RANK"], os.environ["LOCAL_RANK"]
                )
            )
            named_actors = [ray.get_actor("retrieval_worker_{}".format(i)) for i in range(args.num_retrieval_workers)]
    args.actor_handles = named_actors
    assert args.actor_handles == named_actors

    if model is None:
        model: GenerativeQAModule = GenerativeQAModule(args)

    dataset = Path(args.data_dir).name
    if (
        args.logger_name == "default"
        or args.fast_dev_run
        or str(args.output_dir).startswith("/tmp")
        or str(args.output_dir).startswith("/var")
    ):
        training_logger = True  # don't pollute wandb logs unnecessarily
    elif args.logger_name == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        project = os.environ.get("WANDB_PROJECT", dataset)
        training_logger = WandbLogger(name=model.output_dir.name, project=project)

    elif args.logger_name == "wandb_shared":
        from pytorch_lightning.loggers import WandbLogger

        training_logger = WandbLogger(name=model.output_dir.name, project=f"hf_{dataset}")

    es_callback = (
        get_early_stopping_callback(model.val_metric, args.early_stopping_patience)
        if args.early_stopping_patience >= 0
        else False
    )

    trainer: pl.Trainer = generic_train(
        model,
        args,
        logging_callback=Seq2SeqLoggingCallback(),
        checkpoint_callback=get_checkpoint_callback(args.output_dir, model.val_metric),
        early_stopping_callback=es_callback,
        logger=training_logger,
        custom_ddp_plugin=CustomDDP(),
        profiler=pl.profiler.AdvancedProfiler() if args.profile else None,
    )
    pickle_save(model.hparams, model.output_dir / "hparams.pkl")

    if not args.do_predict:
        return model

    # test() without a model tests using the best checkpoint automatically
    trainer.test()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = GenerativeQAModule.add_model_specific_args(parser, os.getcwd())
    parser = GenerativeQAModule.add_retriever_specific_args(parser)
    parser = GenerativeQAModule.add_ray_specific_args(parser)

    # Pytorch Lightning Profiler
    parser.add_argument(
        "--profile",
        action="store_true",
        help="If True, use pytorch_lightning.profiler.AdvancedProfiler to profile the Trainer.",
    )

    args = parser.parse_args()

    main(args)
