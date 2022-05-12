import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel, T5ForConditionalGeneration, T5Config
from models.graph_retriever.KGIN import GraphModel
from transformers import BertModel, BertConfig
from easydict import EasyDict


class KnowledgeGraphRetriever(nn.Module):
    def __init__(self, config, data_config, graph, adj_mat):
        super(KnowledgeGraphRetriever, self).__init__()
        self.config = config
        
        
        QueryEncoderModelClass = globals()[self.config.model_config.QueryEncoderModelClass]

        QueryEncoderConfigClass = globals()[self.config.model_config.QueryEncoderConfigClass]
        query_model_config = QueryEncoderConfigClass.from_pretrained(self.config.model_config.QueryEncoderModelVersion)
        self.query_encoder = QueryEncoderModelClass.from_pretrained(self.config.model_config.QueryEncoderModelVersion, config=query_model_config)
        
        self.graph_model = GraphModel(config, data_config, graph, adj_mat)
        self.projection_layer = nn.Linear(query_model_config.hidden_size, self.config.model_config.dim)

        if self.config.model_config.get('pooling_output', None) is not None:
            self.query_pooler = nn.Sequential(
                nn.Linear(query_model_config.hidden_size, self.config.model_config.pooling_output.dim),
                nn.Dropout(self.config.model_config.pooling_output.dropout)
            )
        else:
            self.query_pooler = None

        

    def resize_token_embeddings(self, dim):
        self.query_encoder.resize_token_embeddings(dim)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        item_input_ids=None,
        item_attention_mask=None,
        labels=None,
        question_indices=None,
        pos_items=None,
        neg_items=None,
        **kwargs
    ):
        # query encoder
        query_outputs = self.query_encoder(input_ids=input_ids,
                                            attention_mask=attention_mask)
        query_last_hidden_states = query_outputs.last_hidden_state
        if self.query_pooler is not None:
            query_last_hidden_states = self.query_pooler(query_last_hidden_states)
        query_embeddings = query_last_hidden_states[:, 0]
        query_embeddings = self.projection_layer(query_embeddings)
        #print('query_embeddings', query_embeddings.shape)

        graph_batch = {
            'users': question_indices,
            'pos_items': pos_items,
            'neg_items': neg_items,
            'query_embeddings': query_embeddings,
        }

        loss, mf_loss, emb_loss, cor = self.graph_model(graph_batch)

        return {
            'loss': loss,
        }


    def generate_query_embeddings(self,
        input_ids=None,
        attention_mask=None,
        question_indices=None,
        **kwargs):
        
         # query encoder
        query_outputs = self.query_encoder(input_ids=input_ids,
                                            attention_mask=attention_mask)
        query_last_hidden_states = query_outputs.last_hidden_state
        if self.query_pooler is not None:
            query_last_hidden_states = self.query_pooler(query_last_hidden_states)
        query_embeddings = query_last_hidden_states[:, 0]
        query_embeddings = self.projection_layer(query_embeddings)
        #print('query_embeddings', query_embeddings.shape)

        graph_batch = {
            'users': question_indices,
            'query_embeddings': query_embeddings,
        }

        updated_query_embneddings, entity_embeddings  = self.graph_model.generate_query_embeddings(graph_batch)
        
        return EasyDict(
            query_embeddings=updated_query_embneddings,
            entity_embeddings=entity_embeddings)

    def load_pretrained_embeddings(self, pretrained_entity_emb):
        self.graph_model.load_pretrained_embeddings(pretrained_entity_emb)