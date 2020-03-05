afrom collections import OrderedDict
from typing import List, Dict, OrderedDict
import json

from transformers import BertTokenizer
from tqdm import tqdm

from data_utils import Entity, Relation, Article, EntityType, RelationType
from data import ERDataset

class ReadInput:
    def __init__(self, path, tokenizer: BertTokenizer, n_count, max_sent_size):
        
        data = json.load(open(path), object_pairs_hook=OrderedDict)
        self.entity_dict = OrderedDict()
        self.relation_dict = OrderedDict()
        self.entity_types = OrderedDict()
        self.relation_types = OrderedDict()
        self.entity_types["None"] = EntityType("None", 0, "None")
        self.entity_dict[0] = self.entity_types['None']
        self.relation_types['None'] = RelationType("None", 0, "Negative")
        
        
        