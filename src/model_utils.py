import os
import sys
import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn
import transformers
from transformers import BertModel, BertTokenizer, BertConfig, BertConfig, BertPreTrainedModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from utils import get_conembedding, get_batch_index

class AscModel(BertPreTrainedModel):
    
    def __init__(self, num_labels, config, sentences):
        super(AscModel, self).__init__(config)
        self.num_labels = num_labels
        self.model = BertModel(config)
        self.input = sentences
        
    # Incomplete
        

class ERModel(BertPreTrainedModel):
    
    def __init__(self, embedding_size, relations, entities, cls_token, config, freeze_transformer):
        super(ERModel, self).__init__(config)
        self.relations = relations
        self.entities = entities
        self.cls_token = cls_token
        self.embedding_size = nn.Embedding(100, embedding_size)
        self.model = BertModel(config)
        self.init_weights()
        self.entity_clf = nn.Linear(config.hidden_size + embedding_size, entities)
        self.relations_clf = nn.Linear(config.hidden_size + embedding_size, relations)
        if freeze_transformer:
            for p in self.model.parameters():
                p.requires_grad = False
        
    def entity_classifier(self, idxs, entity_mask, embedding_size, h):
        entity_span = entity_mask.unsqueeze(-1) * h.unsqueeze(1).repeat(1, entity_mask.shape[1], 1, 1)
        entity_rep = get_conembedding(self.cls_token, h, idxs)
        entity_rep = torch.cat([entity_rep.unsqueeze(1).repeat(1, entity_span.sum(dim=2)[0].shape[1], 1), entity_span.sum(dim=2)[0], embedding_size], dim=2)
        entity_clf = self.entity_clf(entity_rep)
        return entity_clf, entity_span.sum(dim = 2)[0]
    
    def relation_classifier(self, entity_span, relations, relation_mask, h_rel, embedding_size, start):
        pairs = torch.stack([entity_span[i][relations[i]] for i in range(relations.shape[0])])
        pair_embeddings = torch.stack([embedding_size[i][relations[i]] for i in range(relations.shape[0])])
        pairs = pairs.view(relations.shape[0], pairs.shape[1], -1)
        pair_embeddings = pair_embeddings.view(relations.shape[0], pair_embeddings.shape[1], -1)
        relation_rep = relation_mask * h_rel
        relation_rep = relation_rep.sum(dim=2)
        relation_rep = torch.cat([relation_rep[0], pairs, pair_embeddings], dim=2)
        return self.relations_clf(relation_rep)

    def forward_train(self, idxs, entity_mask, entity_size, relations, relation_mask, attn_mask):
        h = self.model(input_ids = idxs, attention_mask =attn_mask.float())[0]
        batch_size = idxs.shape[0]
        embedding_size = self.embedding_size(entity_size)
        entity_res, entity_span = self.entity_classifier(idxs, entity_mask.float(), embedding_size, h)
        h_rel = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], 5),1),1,1)
        relc = torch.zeros([batch_size, relations.shape[1], self.relations]).to(self.relations_clf.weight.device)
        for rel in range(0, relations.shape[1], 5):
            rel_rep_logs = self.relation_classifier(entity_span, relations, relation_mask.float().unsqueeze(-1), h_rel, embedding_size, rel)
            relc[:, rel:rel + 5, :] = rel_rep_logs
        
        return entity_res, relc
    
    
class ERLoss():
    def __init__(self, network, optimizer, schedule,
                 ent_cr, rel_cr, max_gradnorm):
        self.optim = optimizer
        self.model = network
        self.scheduler = schedule
        self.max_grad_norm = max_gradnorm
        self.relc = rel_cr
        self.ec = ent_cr
        
    def compute(self, entity_so, relation_so, entities, relations, entity_mask, relation_mask):
        entities = entities.view(-1)
        relations = relations.view(-1)
        entity_so = entity_so.view(-1, entity_so.shape[-1]).float()
        relation_so = relation_so.view(-1, relation_so.shape[-1]).float()
        entity_mask = entity_mask.view(-1).float()
        relation_mask = relation_mask.view(-1).float()
        
        entity_loss = self.ec(entity_so, entities) 
        entity_loss = (entity_loss * entity_mask).sum() / entity_mask.sum()
        relation_loss = self.relc(relation_so, relations)
        relation_loss = relation_loss.sum(-1) / relation_loss.shape[-1]
        relation_loss = (relation_loss * relation_mask).sum() / relation_mask.sum()
        loss = 0.0
        if torch.isnan(relation_loss):
            loss = entity_loss
        else:
            loss = entity_loss + relation_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optim.step()
        self.scheduler.step()
        self.model.zero_grad()
        
        return loss.item()