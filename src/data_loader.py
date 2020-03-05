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
        self.dataset = dict()
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.entity_dict = OrderedDict()
        self.relation_dict = OrderedDict()
        self.entity_types = OrderedDict()
        self.relation_types = OrderedDict()
        self.entity_types["None"] = EntityType("None", 0, "None")
        self.entity_dict[0] = self.entity_types['None']
        self.relation_types['None'] = RelationType("None", 0, "Negative")
        self.relation_dict[0] = self.relation_types['None']
        self.n_count = n_count
        self.max_sent_size = max_sent_size
        self.context_size = -1
        
        for index, (k, v) in enumerate(data['entities'].items()):
            self.entity_types[k] = EntityType(k, index + 1, v['eid'])
            self.entity_dict[index + 1] = self.entity_types[k]
            
        for index, (k, v) in enumerate(data['relations'].items()):
            self.relation_types[k] = RelationType(k, index + 1, v['rel_id'])
            self.relation_dict = self.relation_types[k]
            
        self.len_entity_types = len(entity_types)
        self.len_relation_types = len(relation_types)
            
    def process_tokens(self, input_tokens, dataset : ERDataset):
        encoding = list(self.tokenizer.convert_tokens_to_ids('[CLS]'))
        tokens = list()
        
        for index, string in enumerate(input_tokens):
            t_encoding = self.tokenizer.encode(string)
            encoding += t_encoding
            start, end = tuple(len(encoding), len(encoding) + len(t_encoding))
            tokens.append(dataset.get_new_token(start, end, index, string))
        encoding += list(self.tokenizer.convert_tokens_to_ids('[SEP]'))
        
        return tokens, encoding
    
    
    def process_entities(self, input_entities, tokens, dataset: ERDataset):
        entities = list()
        for _, entity in enumerate(input_entities):
            e_type = self.entity_types[input_entities[entity['type']]]
            start, end = tuple(entity['start'], entity['end'])
            string = " ".join([token.string for token in tokens[start:end]])
            entities.append(dataset.get_new_entity(e_type, tokens[start:end], string))
        
        return entities
        
    def process_relations(self, input_relations, entities, dataset : ERDataset):
        relations = list()
        for relation in input_relations:
            r_type = self.relation_types[relation['type']]
            first_e = entities[relation['head']]
            second_e = entities[relation['tail']]
            relations.append(dataset.get_new_relation(r_type, first_e, second_e))
            
        return relations    
        