from collections import OrderedDict
from typing import Dict, List

from torch.utils.data import Dataset

from data_utils import Iterator, Token, Span, EntityType, Entity, RelationType, Relation, Article
from utils import create_asc_masks, create_entites_mask, create_relations_mask

class ERDataset(Dataset):
    def __init__(self, label, i, relation_types, entity_types, num_relations, num_entites):
        self.mode_map = {1:'TRAIN', 2:'TEST'}
        self.mode = 1
        self.token_id = 0
        self.label = label
        self.input = i
        self.e_id = 0
        self.entities = OrderedDict()
        self.rel_id = 0
        self.relations = OrderedDict()
        self.pmid = 0
        self.max_size = 20
        self.doc = OrderedDict()
        self.entity_types = entity_types
        self.relation_types = relation_types
        self.num_entites = num_entites
        self.num_relations = num_relations
        
    def change_mode(self, mode):
        if ((mode == 1 and self.mode == 1) or (mode == 2 and self.mode == 2)):
            return
        self.mode = 2
        
    def get_new_token(self, start, end, idn, string):
        new_token = Token(start, end, idn, self.token_id, string)
        self.token_id += 1
        return new_token
        
    def get_new_entity(self, etype, tokens, string):
        self.entities[self.e_id] = Entity(etype, tokens, string, self.e_id)
        self.e_id += 1
        return self.entities[self.e_id - 1]
        
    def get_new_relation(self, rtype, first_e, second_e):
        self.relations[self.rel_id] = Relation(rtype, self.rel_id, first_e, second_e)
        self.rel_id += 1
        return self.relations[self.rel_id - 1]
        
    def get_new_doc(self, assigned_id, encoding, entities, relations, tokens):
        self.doc[self.pmid] = Article(self.pmid, encoding, entities, relations, tokens)
        self.pmid += 1
        return self.doc[self.pmid - 1]
    
    def get_training_sample(self, doc_id, num_relations, max_size):
        e_spans = list()
        r_spans = list()
        e_types = list()
        r_types = list()
        e_types_ix = list()
        e_masks = list()
        r_masks = list()
        e_size = list()
        for entity in self.doc[doc_id].entities:
            e_types.append(entity.e_type.string)
            e_types_ix.append(entity.e_type.index)
            e_spans.append(entity.span)
            masks = create_entites_mask(*entity.span, len(self.doc[doc_id].encoding))
            e_masks.append(masks)
            e_size.append(len(entity))
        for relation in self.doc[doc_id].relations:
            es1 = relation.first_entity.span
            es2 = relation.second_entity.span
            
            
            
            
        
    def itr_rel(self, batch_size, order):
        rel_itr = Iterator(batch_size, self.relations, order)
        return rel_itr
        
    def itr_articles(self, batch_size, order):
        art_itr = Iterator(batch_size, self.doc, order)
        return art_itr
    
    def __len__(self):
        return len(self.doc)
        
    
        
        