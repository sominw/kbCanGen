import os
from collections import OrderedDict
from math import ceil, floor
import argparse

import torch
from torch.optim import Optimizer
from torch.nn import DataParallel
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm, trange

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
        data = reader(path, self.tokenizer, self.args.max_span_size, self.args.ne_count, self.args.nr_count)
        data.read_data(temp)
        