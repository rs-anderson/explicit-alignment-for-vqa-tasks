from enum import Enum

import torch
import torch.nn as nn
from torch.nn import functional as nnf

from transformers import (
    AutoModelForSeq2SeqLM,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
    LogitsProcessorList,
    MaxLengthCriteria,
    StoppingCriteriaList,
    get_linear_schedule_with_warmup,
)
from flamingo_pytorch import PerceiverResampler

import logging

logger = logging.getLogger(__name__)

from typing import Tuple, Optional, Union

import argparse
import json

import os
from einops import rearrange, repeat

import types

from transformers.modeling_outputs import BaseModelOutput

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MappingType(Enum):
    MLP = "mlp"
    Transformer = "transformer"

class PerceiverResamplerForSingleImage(PerceiverResampler):
    def __init__(
        self,
        *,
        num_latents,
        latents_init,
        **kwargs
    ):
        super().__init__(num_latents = num_latents, **kwargs)

    # use default initialisation of latents
    def forward(self, x):
        x = x.unsqueeze(2)
        return super().forward(x)


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


def prepare_inputs_for_generation(
    self,
    input_ids,
    **kwargs
):  

    if input_ids.shape[1] != 1:
        decoder_attention_mask = kwargs["decoder_attention_mask"]
        kwargs["decoder_attention_mask"] = torch.cat(
            [
                decoder_attention_mask,
                decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], input_ids.shape[1]-1)),
            ],
            dim=-1,
        )
        decoder_inputs_embeds = kwargs["decoder_inputs_embeds"]
        embedding_of_new_input = self.shared(input_ids[:, 1:])
        kwargs["decoder_inputs_embeds"] = torch.cat(
            [
                decoder_inputs_embeds,
                embedding_of_new_input,
            ],
            dim=1,
        )
    return {
        "inputs_embeds": kwargs.get('inputs_embeds', None),
        "attention_mask": kwargs.get('attention_mask', None),
        "decoder_inputs_embeds": kwargs.get('decoder_inputs_embeds', None),
        "decoder_attention_mask": kwargs.get('decoder_attention_mask', None),
        "encoder_outputs": kwargs.get('encoder_outputs', None),
        "use_cache": kwargs.get('use_cache', None),
    }


class VCT0Model(nn.Module):
    def __init__(
        self,
        prefix_length: int,
        clip_length: Optional[int] = None,
        prefix_size: int = 512,
        num_layers: int = 8,
        mapping_type: str = "mlp",
        model_version: str = "bigscience/T0_3B",
    ):
        super(VCT0Model, self).__init__()
        self.prefix_length = prefix_length
        self.lm = AutoModelForSeq2SeqLM.from_pretrained(model_version)
        self.lm_embedding_size = self.lm.model_dim
        if mapping_type == "mlp":
            print("\n\n Using MLP \n\n")
            self.clip_project = MLP(
                (
                    prefix_size,
                    (self.lm_embedding_size * prefix_length) // 2,
                    self.lm_embedding_size * prefix_length,
                )
            )
        elif mapping_type == "transformer":
            print("\n\n Using Transformer \n\n")
            self.clip_project = TransformerMapper(
                prefix_size,
                self.lm_embedding_size,
                prefix_length,
                clip_length,
                num_layers,
            )
        elif mapping_type == "perceiver":
            print("\n\n Using Perceiver \n\n")
            latents_init = self.sample_init_embeddings_from_vocab(vocab_size=32128, dim=prefix_length)

            self.clip_project = PerceiverResamplerForSingleImage(
                dim = self.lm_embedding_size,
                depth = 2,
                dim_head = 64,
                heads = 8,
                num_latents = prefix_length,    # the number of latents to shrink your media sequence to, perceiver style
                num_time_embeds = 1,
                ff_mult = 1,
                latents_init = latents_init,
            )
        else:
            print("\n\n Unrecognised mapping type \n\n")
            print("\n\n Setting mapping type to MLP \n\n")
            self.clip_project = MLP(
                (
                    prefix_size,
                    (self.lm_embedding_size * prefix_length) // 2,
                    self.lm_embedding_size * prefix_length,
                )
            )
        print(self.clip_project)
    
    def sample_init_embeddings_from_vocab(self, vocab_size, dim):
        idx = torch.randint(0, vocab_size, (dim,))
        print(f"Input tokens used to initialise latents are: {idx}")
        return self.lm.shared(idx).detach().clone()

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
        prefix: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):

        prefix_projections = self.clip_project(prefix).view(
            -1, self.prefix_length, self.lm_embedding_size
        )

        out = self.lm(
            inputs_embeds=prefix_projections,
            labels=labels,
        )
        return out

    def generate(
        self,
        prefix: torch.Tensor,
        question_tokens: Optional[torch.Tensor] = None,
        question_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        no_prefix: Optional[bool] = False,
        pass_examples_through_encoder_one_at_a_time: Optional[bool] = False,
        num_shots: Optional[int] = None,
        **generation_kwargs,
    ):
        if no_prefix:

            if pass_examples_through_encoder_one_at_a_time:
                batch_size = prefix.shape[0]
                # pass each in-context example through encoder one-by-one. Then cat and pass to decoder via 'encoder_ouputs'
                outputs_list = []
                for i in range(batch_size):
                    encoder_outputs = self.lm.encoder(input_ids=question_tokens[i], attention_mask=question_mask[i]).last_hidden_state
                    encoder_outputs_list = [encoder_output for encoder_output in encoder_outputs]
                    encoder_outputs_all = BaseModelOutput(last_hidden_state=torch.cat(encoder_outputs_list).unsqueeze(0))
                    outputs_list.append(self.lm.generate(encoder_outputs=encoder_outputs_all), **generation_kwargs)
                
                return outputs_list
            
            else:
                return self.lm.generate(
                    inputs=question_tokens, attention_mask=question_mask, **generation_kwargs
                )

        if pass_examples_through_encoder_one_at_a_time:
            batch_size = question_tokens.shape[0]
            num_shots = prefix.shape[1] -1
            embedding_text = self.lm.shared(question_tokens)
            prefix_projections = self.clip_project(prefix).view(
                batch_size, -1, self.prefix_length, self.lm_embedding_size
            )

            encoder_outputs_list = []
            joint_attention_masks_list = []
            for i in range(num_shots+1):
                joint_embeddings_for_example, joint_attention_masks_for_example = self.insert_prefix_into_input(batch_size, 0, question_tokens[:, i], embedding_text[:, i], prefix_projections[:, i].contiguous(), question_mask[:, i], special_token_id=32099-i)
                encoder_outputs_for_example = self.lm.encoder(inputs_embeds=joint_embeddings_for_example, attention_mask=joint_attention_masks_for_example).last_hidden_state
                encoder_outputs_list.append(encoder_outputs_for_example)
                joint_attention_masks_list.append(joint_attention_masks_for_example)
                
            encoder_outputs = BaseModelOutput(last_hidden_state=torch.cat(encoder_outputs_list, dim=1))
            return self.lm.generate(encoder_outputs=encoder_outputs, attention_mask=torch.cat(joint_attention_masks_list, dim=1), **generation_kwargs)

        if question_tokens is not None:
            batch_size = question_tokens.shape[0]
            embedding_text = self.lm.shared(question_tokens)

            prefix_projections = self.clip_project(prefix).view(
                batch_size, -1, self.prefix_length, self.lm_embedding_size
            )
            num_shots = prefix.shape[1] - 1 if not num_shots else num_shots
            
            if decoder_input_ids is None:
                joint_embeddings, joint_attention_masks = self.insert_prefix_into_input(batch_size, num_shots, question_tokens, embedding_text, prefix_projections, question_mask)
            
                input_length = joint_attention_masks.shape[1]
                if input_length > 1024:
                    logger.warning(f"input length {input_length} is greater than 1024! \n\n")
                
                return self.lm.generate(
                    inputs_embeds=joint_embeddings, attention_mask=joint_attention_masks, **generation_kwargs
                )
            
            else:
                # self.lm.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation, self.lm)
                # decoder_embedding_text = self.lm.shared(decoder_input_ids)
                joint_embeddings, joint_attention_masks = self.insert_prefix_into_input(batch_size, 0, question_tokens, embedding_text, prefix_projections[:, -1].contiguous(), question_mask)
                # joint_decoder_embeddings, joint_decoder_attention_masks = self.insert_prefix_into_input(batch_size, num_shots, decoder_input_ids, decoder_embedding_text, prefix_projections, decoder_attention_mask)

                input_length = joint_attention_masks.shape[1]
                if input_length > 1024:
                    logger.warning(f"input length {input_length} is greater than 1024! \n\n")
                
                outputs = self.lm.generate(
                    inputs_embeds=joint_embeddings, attention_mask=joint_attention_masks, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, **generation_kwargs
                )
                return outputs[:, decoder_input_ids.shape[1]:]
                # return self.lm.generate(
                #     inputs_embeds=joint_embeddings, attention_mask=joint_attention_masks, decoder_inputs_embeds=joint_decoder_embeddings, decoder_attention_mask=joint_decoder_attention_masks, **generation_kwargs
                # )
        
        else:
            prefix_projections = self.clip_project(prefix).view(
                -1, self.prefix_length, self.lm_embedding_size
            )

            return self.lm.generate(
                inputs_embeds=prefix_projections, **generation_kwargs
            )


    def insert_prefix_into_input(
        self, batch_size, num_shots, batch_question_tokens, batch_text_embeddings, batch_prefix_projections, batch_question_masks, special_token_id = 32099
    ):
        num_image_tokens = num_shots + 1
        input_sequence_length = batch_question_tokens.shape[1]
        input_sequence_length_wo_image_tokens = input_sequence_length - num_image_tokens
        output_seq_length = input_sequence_length + (self.prefix_length-1)*num_image_tokens

        embedding_out = torch.ones((batch_size, output_seq_length, self.lm_embedding_size), device=device) * -100
        attention_mask_out = torch.ones((batch_size, output_seq_length), dtype=int, device=device) * -100
        text_tokens_mask = torch.zeros((batch_size, output_seq_length), dtype=int, device=device)

        all_special_token_indices = torch.zeros(batch_question_tokens.shape, dtype=int, device=device)
        
        for i in range(num_shots+1):
            all_special_token_indices += (batch_question_tokens == special_token_id - i)

        cumulative_count_of_indices = torch.cumsum(all_special_token_indices, dim=1)
        cumulative_count_of_indices_without_special_tokens = cumulative_count_of_indices[~all_special_token_indices.bool()].view(batch_size, input_sequence_length_wo_image_tokens)

        text_embedding_row_inds = torch.arange(input_sequence_length_wo_image_tokens, device=device)
        inds_for_batch_text_embeddings_to_keep = text_embedding_row_inds + cumulative_count_of_indices_without_special_tokens*self.prefix_length

        inds_for_text_tokens = (1-all_special_token_indices).bool()
        batch_text_embeddings_to_keep = batch_text_embeddings[inds_for_text_tokens].view(batch_size, input_sequence_length_wo_image_tokens, self.lm_embedding_size)
        batch_question_masks_to_keep = batch_question_masks[inds_for_text_tokens].view(batch_size, input_sequence_length_wo_image_tokens)

        batch_inds_for_broadcasting = [[i] for i in range(batch_size)]
        text_tokens_mask[batch_inds_for_broadcasting, inds_for_batch_text_embeddings_to_keep] = 1

        text_embedding_inds = text_tokens_mask.bool()
        prefix_embedding_inds = ~text_tokens_mask.bool()

        embedding_out[text_embedding_inds] = batch_text_embeddings_to_keep.view(-1, self.lm_embedding_size)
        embedding_out[prefix_embedding_inds] = batch_prefix_projections.view(-1, self.lm_embedding_size)

        attention_mask_out[text_embedding_inds] = batch_question_masks_to_keep.flatten()
        attention_mask_out[prefix_embedding_inds] = 1

        return embedding_out, attention_mask_out

class VCT0Prefix(VCT0Model):
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(VCT0Prefix, self).train(mode)
        for param in self.lm.parameters():
            param.requires_grad = False
        self.lm.eval()
        return self


def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, "w") as outfile:
        json.dump(config, outfile)


