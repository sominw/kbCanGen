from collections import OrderedDict
import json

from transformers import BertTokenizer

from data_utils import Entity, Relation, Article, EntityType, RelationType
from data import ERDataset

class ReadInput:
    def __init__(self, path, tokenizer: BertTokenizer, max_sent_size, num_entites, num_relations):
        
        data = json.load(open(path), object_pairs_hook=OrderedDict)
        self.datasets = dict()
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
        self.num_entites = num_entites
        self.num_relations = num_relations
        self.max_sent_size = max_sent_size
        self.context_size = 0
        
        for index, (k, v) in enumerate(data['entities'].items()):
            self.entity_types[k] = EntityType(k, index + 1, v['eid'])
            self.entity_dict[index + 1] = self.entity_types[k]
            
        for index, (k, v) in enumerate(data['relations'].items()):
            self.relation_types[k] = RelationType(k, index + 1, v['rel_id'])
            self.relation_dict = self.relation_types[k]
            
        self.len_entity_types = len(self.entity_types)
        self.len_relation_types = len(self.relation_types)
        
    def fetch_data(self, label):
        return self.datasets[label]
    
    def fetch_e_type(self, index):
        return self.entity_dict[index]
    
    def fetch_r_type(self, index):
        return self.relation_dict[index]
            
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
    
    def process_documents(self, d, dataset: ERDataset):
        t = d["tokens"]
        e = d["entities"]
        r = d["relations"]
        tokens, encoding = self.process_tokens(t, dataset)
        entities = self.process_entities(e, tokens, dataset)
        relations = self.process_relations(r, entities, dataset)
        doc = dataset.get_new_doc(encoding, entities, relations, tokens)
        return doc
    
    # Input to read data is a dict with train & dev paths.
    def read_data(self, path):
        for label, p in path.items():
            dataset = ERDataset(label, self.relation_types, self.entity_types, self.num_relations, self.num_entites)
            documents = json.load(open(p))
            for document in documents:
                self.process_documents(document, dataset)
            self.datasets[label] = dataset
        temp = list()
        for d in self.datasets:
            for article in d.doc:
                temp.append(len(article.encoding))
        self.context_size = max(temp)
                 