from collections import OrderedDict
from typing import List, Dict, OrderedDict
import json

from transformers import BertTokenizer
from tqdm import tqdm

from data_utils import Entity, Relation, Article, EntityType, RelationType
from data import ERDataset

class ReadInput:
    def __init__(self, path, tokenizer: BertTokenizer, n_count, max_sent_size):
        
        data = json.load(open(path), object_pairs_hook=OrderedDict)
        self.entity_ids = OrderedDict()
        self.relation_ids = OrderedDict()
        self.entity_types = OrderedDict()
        self.relation_types = OrderedDict()