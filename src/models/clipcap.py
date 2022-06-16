from enum import Enum

import torch
import torch.nn as nn
from torch.nn import functional as nnf

from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
    LogitsProcessorList,
    MaxLengthCriteria,
    StoppingCriteriaList,
    get_linear_schedule_with_warmup,
)
from typing import Tuple, Optional, Union

import argparse
import json

import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MappingType(Enum):
    MLP = "mlp"
    Transformer = "transformer"


class MLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class MlpTransformer(nn.Module):
    def __init__(
        self,
        in_dim,
        h_dim,
        out_d: Optional[int] = None,
        act=nnf.relu,
        dropout=0.0,
    ):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim**-0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(
            b, n, self.num_heads, c // self.num_heads
        )
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(
            b, m, 2, self.num_heads, c // self.num_heads
        )
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum("bnhd,bmhd->bnmh", queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum("bnmh,bmhd->bnhd", attention, values).reshape(
            b, n, c
        )
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):
    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(
        self,
        dim_self,
        dim_ref,
        num_heads,
        mlp_ratio=4.0,
        bias=False,
        dropout=0.0,
        act=nnf.relu,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(
            dim_self, dim_ref, num_heads, bias=bias, dropout=dropout
        )
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(
            dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout
        )


class Transformer(nn.Module):
    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec:  # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(
        self,
        dim_self: int,
        num_heads: int,
        num_layers: int,
        dim_ref: Optional[int] = None,
        mlp_ratio: float = 2.0,
        act=nnf.relu,
        norm_layer: nn.Module = nn.LayerNorm,
        enc_dec: bool = False,
    ):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(
                    TransformerLayer(
                        dim_self,
                        dim_ref,
                        num_heads,
                        mlp_ratio,
                        act=act,
                        norm_layer=norm_layer,
                    )
                )
            elif enc_dec:  # self
                layers.append(
                    TransformerLayer(
                        dim_self,
                        dim_self,
                        num_heads,
                        mlp_ratio,
                        act=act,
                        norm_layer=norm_layer,
                    )
                )
            else:  # self or cross
                layers.append(
                    TransformerLayer(
                        dim_self,
                        dim_ref,
                        num_heads,
                        mlp_ratio,
                        act=act,
                        norm_layer=norm_layer,
                    )
                )
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):
    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(
            x.shape[0], *self.prefix_const.shape
        )
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length :]
        return out

    def __init__(
        self,
        dim_clip: int,
        dim_embedding: int,
        prefix_length: int,
        clip_length: int,
        num_layers: int = 8,
    ):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(
            torch.randn(prefix_length, dim_embedding), requires_grad=True
        )


class ClipCaptionModel(nn.Module):
    def __init__(
        self,
        prefix_length: int,
        clip_length: Optional[int] = None,
        prefix_size: int = 512,
        num_layers: int = 8,
        mapping_type: str = "mlp",
        model_version: str = "gpt2",
    ):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained(model_version)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == "mlp":
            print("\n\n Using MLP \n\n")
            self.clip_project = MLP(
                (
                    prefix_size,
                    (self.gpt_embedding_size * prefix_length) // 2,
                    self.gpt_embedding_size * prefix_length,
                )
            )
        else:
            print("\n\n Using Transformer \n\n")
            self.clip_project = TransformerMapper(
                prefix_size,
                self.gpt_embedding_size,
                prefix_length,
                clip_length,
                num_layers,
            )
        print(self.clip_project)

    def get_dummy_token(
        self,
        batch_size: int,
        num_question_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        return (
            torch.ones(
                batch_size,
                self.prefix_length + num_question_tokens,
                dtype=torch.int64,
                device=device,
            )
            * -100
        )

    def forward(
        self,
        question_tokens: torch.Tensor,
        # answer_tokens: torch.Tensor,
        prefix: torch.Tensor,
        question_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pad_token_id: Optional[torch.Tensor] = None,
    ):
        batch_size = question_tokens.shape[0]
        num_question_tokens = question_tokens.shape[1]
        # num_answer_tokens = answer_tokens.shape[1]

        prefix_mask = torch.ones(
            (batch_size, self.prefix_length),
            device=device,
        )

        input_tokens = question_tokens

        attention_mask = torch.cat(
            (
                prefix_mask,
                question_mask,
            ),
            dim=1,
        )
        embedding_text = self.gpt.transformer.wte(input_tokens)
        prefix_projections = self.clip_project(prefix).view(
            -1, self.prefix_length, self.gpt_embedding_size
        )
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)

        labels = torch.cat(
            (
                torch.ones(
                    batch_size,
                    self.prefix_length,
                    dtype=torch.int64,
                    device=device,
                )
                * -100,
                labels,
            ),
            dim=1,
        )

        out = self.gpt(
            inputs_embeds=embedding_cat,
            labels=labels,
            attention_mask=attention_mask,
        )
        return out

    def generate(
        self,
        question_tokens: torch.Tensor,
        prefix: torch.Tensor,
        question_mask: Optional[torch.Tensor] = None,
        **generation_kwargs,
    ):
        batch_size = question_tokens.shape[0]

        prefix_mask = torch.ones(
            (batch_size, self.prefix_length), device=device
        )
        # attention_mask = []
        # num_zero_attentions = (question_mask == 0).sum(dim=1)
        # for i in range(batch_size):
        #     individual_question_mask = question_mask[i]
        #     individual_attention_mask = torch.cat(
        #         (
        #             torch.zeros((1, num_zero_attentions[i]), device=device),
        #             prefix_mask[i].unsqueeze(dim=0),
        #             individual_question_mask[individual_question_mask != 0].unsqueeze(dim=0),
        #         ),
        #         dim=1,
        #     )
        #     attention_mask.append(individual_attention_mask)
        attention_mask = torch.cat(
            (
                prefix_mask,
                question_mask,
            ),
            dim=1,
        )

        embedding_text = self.gpt.transformer.wte(question_tokens)
        prefix_projections = self.clip_project(prefix).view(
            -1, self.prefix_length, self.gpt_embedding_size
        )
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)

        return self._generate_from_embeddings(
            embedding_cat, attention_mask, **generation_kwargs
        )

    def _generate_from_embeddings(
        self,
        embedding_cat,
        attention_mask,
        max_length: Optional[int] = 10,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ):

        tokens = None

        pad_token_id = (
            pad_token_id
            if pad_token_id is not None
            else self.gpt.config.pad_token_id
        )
        eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else self.gpt.config.eos_token_id
        )
        unfinished_sequences = (
            embedding_cat.new(embedding_cat.shape[0]).fill_(1).unsqueeze(1)
        )
        cur_len = embedding_cat.shape[-1]

        generated_input_embeds = embedding_cat
        for i in range(max_length):

            outputs = self.gpt(
                inputs_embeds=generated_input_embeds,
                attention_mask=attention_mask,
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, -1).unsqueeze(1)

            next_token_embed = self.gpt.transformer.wte(next_tokens)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                    )
                next_tokens = (
                    next_tokens * unfinished_sequences
                    + pad_token_id * (1 - unfinished_sequences)
                )
            if tokens is None:
                tokens = next_tokens
            else:
                tokens = torch.cat((tokens, next_tokens), dim=1)

            generated_input_embeds = torch.cat(
                (generated_input_embeds, next_token_embed), dim=1
            )

            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        device=device,
                    ),
                ],
                dim=-1,
            )

            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    (next_tokens != eos_token_id).long()
                )

            if unfinished_sequences.max() == 0:
                break

            # if eos_token_id == next_token:
            #     break

        token_list = tokens.cpu().numpy().astype(int).tolist()

        return token_list

    def greedy_search(self, input_ids, logits_processor, stopping_criteria):
        # init values
        logits_processor = (
            logits_processor
            if logits_processor is not None
            else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria
            if stopping_criteria is not None
            else StoppingCriteriaList()
        )
        pad_token_id = (
            pad_token_id
            if pad_token_id is not None
            else self.gpt.config.pad_token_id
        )
        eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else self.gpt.config.eos_token_id
        )
        output_scores = (
            output_scores
            if output_scores is not None
            else self.gpt.config.output_scores
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.gpt.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.gpt.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.gpt.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        cross_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]

        while True:

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(  # TODO: update this function. Maybe just directly insert model inputs into greedy search function?
                input_ids, **model_kwargs
            )

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                    )
                next_tokens = (
                    next_tokens * unfinished_sequences
                    + pad_token_id * (1 - unfinished_sequences)
                )

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    (next_tokens != eos_token_id).long()
                )

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(
                input_ids, scores
            ):
                break

            return input_ids


class ClipCaptionPrefix(ClipCaptionModel):
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        for param in self.gpt.parameters():
            param.requires_grad = False
        self.gpt.eval()
        return self


def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, "w") as outfile:
        json.dump(config, outfile)


def load_model(config_path: str, epoch_or_latest: Union[str, int] = "_latest"):
    with open(config_path) as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(
        args.out_dir, f"{args.prefix}{epoch_or_latest}.pt"
    )
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length)
    else:
        model = ClipCaptionModel(args.prefix_length)
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
    else:
        print(f"{model_path} is not exist")
    return model, parser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data/coco/oscar_split_train.pkl")
    parser.add_argument("--out_dir", default="./checkpoints")
    parser.add_argument(
        "--prefix", default="coco_prefix", help="prefix for saved filenames"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--prefix_length", type=int, default=10)
    parser.add_argument("--prefix_length_clip", type=int, default=10)
    parser.add_argument("--bs", type=int, default=40)
    parser.add_argument(
        "--only_prefix", dest="only_prefix", action="store_true"
    )
    parser.add_argument(
        "--mapping_type", type=str, default="mlp", help="mlp/transformer"
    )
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--is_rn", dest="is_rn", action="store_true")
    parser.add_argument(
        "--normalize_prefix", dest="normalize_prefix", action="store_true"
    )
    args = parser.parse_args()
    prefix_length = args.prefix_length
    # dataset = ClipCocoDataset(args.data, prefix_length, normalize_prefix=args.normalize_prefix)
    prefix_dim = 640 if args.is_rn else 512
    args.mapping_type = {
        "mlp": MappingType.MLP,
        "transformer": MappingType.Transformer,
    }[args.mapping_type]
    if args.only_prefix:
        model = ClipCaptionPrefix(
            prefix_length,
            clip_length=args.prefix_length_clip,
            prefix_size=prefix_dim,
            num_layers=args.num_layers,
            mapping_type=args.mapping_type,
        )
        print("Train only prefix")
    else:
        model = ClipCaptionModel(
            prefix_length,
            clip_length=args.prefix_length_clip,
            prefix_size=prefix_dim,
            num_layers=args.num_layers,
            mapping_type=args.mapping_type,
        )
        print("Train both prefix and GPT")
        # sys.stdout.flush()
    # train(dataset, model, args, output_dir=args.out_dir, output_prefix=args.prefix)


if __name__ == "__main__":
    main()
