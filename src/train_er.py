import os
from collections import OrderedDict
from math import ceil, floor
import argparse

import torch
from torch.optim import Optimizer
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler

import transformers
from transformers import AdamW, BertTokenizer

from data_loader import ReadInput
from data import ERDataset
from model_utils import ERModel, ERLoss

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path, do_lower_case = args.convert_to_lowercase)
        
    def train(self, train_path, val_path, path, reader : ReadInput):
        temp = {
            'train':train_path,
            'val':val_path,
        }
        args = self.args
        data = reader(path, self.tokenizer, self.args.max_span_size, self.args.ne_count, self.args.nr_count)
        data.read_data(temp)
        training_data = data.fetch_data('train')
        val_data = data.fetch_data('val')
        updates = (len(training_data) // args.batch_size) * args.epochs
        model = ERModel.from_pretrained(args.pretrained_path,
                                        embedding_size = args.embedding_size,
                                        cls_token = self.tokenizer.convert_tokens_to_ids('[CLS]'),
                                        freeze_transformer=args.freeze_model_layers,
                                        entities = len(reader.entity_types),
                                        relations = len(reader.relation_types) - 1)
        model.to(self.device)
        optim = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = transformers.get_linear_schedule_with_warmup(optim, args.lr_warmup * updates, updates)
        e_cr = torch.nn.BCEWithLogitsLoss(reduction="None")
        r_cr = torch.nn.BCEWithLogitsLoss(reduction="None")
        loss = ERLoss(model, optim, scheduler, e_cr, r_cr, args.max_grad_norm)
        
        ######## Write Train Epoch & Eval ############
        
        for epoch in range(args.epochs):
            self._train(model, loss, optim, training_data, )
        
    def _train(self, model, loss, optim, data):
        args = self.args
        dl = DataLoader(data, batch_size=args.batch_size, shuffle=True, drop_last=True)
        model.zero_grad()
        iter = 0
        for _, sample in enumerate(dl):
            sample = sample.to(self.device)
            l_e, l_r = model(idxs = sample['encodings'],
                             relations = sample['relations'],
                             entity_size = sample['entity_sizes'],
                             entity_mask = sample['entity_masks'],
                             relation_mask = sample['relation_masks'],
                             attn_mask = sample['context_masks'])
            net_loss = loss.compute(l_e, l_r, sample['entity_types'], sample['relation_types'], sample['entity_masks'], sample['relation_masks'])
            iter += 1
        return iter