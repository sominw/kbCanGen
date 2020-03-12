from collections import OrderedDict
from random import sample

import torch
from torch.utils.data import Dataset

from data_utils import Iterator, Token, Span, EntityType, Entity, RelationType, Relation, Article
from utils import create_asc_masks, create_entites_mask, create_relations_mask
class ERDataset(Dataset):
    def __init__(self, label, relation_types, entity_types, num_relations, num_entites):
        self.mode_map = {1:'TRAIN', 2:'TEST'}
        self.mode = 1
        self.token_id = 0
        self.label = label
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
        self.mode = mode
        
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
        
    def get_new_doc(self, encoding, entities, relations, tokens):
        self.doc[self.pmid] = Article(self.pmid, encoding, entities, relations, tokens)
        self.pmid += 1
        return self.doc[self.pmid - 1]
    
    def get_training_sample(self, doc_id, n_count, max_size):
        
        encodings = torch.tensor(self.doc[doc_id].encoding, dtype=torch.long)
        masks = torch.ones(len(self.doc[doc_id].encoding), dtype=torch.long)
        
        e_spans = list()
        non_e_spans = list()
        r_spans = list()
        non_r_spans = list()
        e_types = list()
        r_types = list()
        non_r_types = list()
        e_types_ix = list()
        e_masks = list()
        non_e_masks = list()
        r_masks = list()
        non_r_masks = list()
        rel = list()
        n_rel = list()
        e_size = list()
        non_e_size = list()
        
        for entity in self.doc[doc_id].entities:
            e_types.append(entity.e_type.string)
            e_types_ix.append(entity.e_type.index)
            e_spans.append(entity.span)
            masks = create_entites_mask(*entity.span, len(self.doc[doc_id].encoding))
            e_masks.append(masks)
            e_size.append(len(entity))
        
        for block in range(1, self.max_size + 1):
            for i in range((len(self.doc[doc_id].tokens) - block) + 1):
                curr = self.doc[doc_id].tokens[i:i + block].astuple()
                if curr not in e_spans:
                    non_e_spans.append(curr)
                    non_e_size.append(block)
        
        for relation in self.doc[doc_id].relations:
            es1 = relation.first_entity.span
            es2 = relation.second_entity.span
            r_spans.append((es1, es2))
            r_types.append(relation.rel_type)
            masks = create_relations_mask(es1, es2, len(self.doc[doc_id].encoding))
            r_masks.append(masks)
            rel.append((e_spans.index(es1)),(e_spans.index(es2)))
            
        for _, es1 in enumerate(e_spans):
            for _, es2 in enumerate(e_spans):
                if (es1, es2) not in e_spans and (es2, es1) not in e_spans and es1 != es2:
                    non_r_spans.append((es1, es2))
                    
        non_e = sample(list(zip(non_e_spans, non_e_size)), n_count)
        non_e_spans, non_e_size = zip(*non_e) if non_e else tuple(list(), list())
        non_e_types = [0] * len(non_e_spans)
        non_e_size = list(non_e_size)
        for s in non_e_spans:
            non_e_masks.append(create_entites_mask(*s, len(self.doc[doc_id].encoding)))
        non_r_spans = sample(non_r_spans, n_count)
        
        for es1, es2 in non_r_spans:
            n_rel.append((e_spans.index(es1), e_spans.index(es2)))
        for s in non_r_spans:
            non_r_masks.append(create_relations_mask(*s, len(self.doc[doc_id].encoding)))
        non_r_types = [0] * len(non_r_spans)
        
        entity_types = torch.tensor(e_types + non_e_types, dtype=torch.long)
        relation_types = torch.tensor([relation.index for relation in r_types] + non_r_types, dtype=torch.long)
        relation_types = torch.zeros([relation_types.shape[0], 2], dtype=torch.float)
        relation_types = relation_types.scatter(1, relation_types.unsqueeze(1),1)[:,1:]
        entity_masks = torch.stack(e_masks + non_e_masks)
        relation_masks = torch.stack(r_masks + non_r_masks)
        entity_size = torch.tensor(e_size + non_e_size, dtype=torch.long)
        relations = torch.tensor(rel + n_rel, dtype=torch.long)
        entity_sample_mask = torch.ones(list(entity_masks.shape[0]), dtype=torch.bool)
        relation_sample_mask = torch.ones(list(relations.shape[0]), dtype=torch.bool)
        
        training_sample = {
            "context_masks":masks,
            "entity_masks":entity_masks,
            "relation_masks":relation_masks,
            "entity_types":entity_types,
            "relation_types":relation_types,
            "relations":relations,
            "entity_sample_masks":entity_sample_mask,
            "relation_sample_masks":relation_sample_mask,
            "entity_sizes":entity_size,
            "encodings":encodings,
        }
        
        return training_sample
        

    def get_test_sample(self, doc_id):
        
        encodings = torch.zeros(len(self.doc[doc_id].encoding), dtype=torch.long)
        encodings[:len(self.doc[doc_id].encoding)] = torch.tensor(self.doc[doc_id].encoding, dtype=torch.long)
        masks = torch.zeros(len(self.doc[doc_id].encoding), dtype=torch.bool)
        masks[:len(self.doc[doc_id].encoding)] = 1
        
        e_size = list()
        e_spans = list()
        e_masks = list()
        
        for block in range(1, self.max_size + 1):
            for i in range((len(self.doc[doc_id].tokens) - block) + 1):
                curr = self.doc[doc_id].tokens[i:i + block].astuple()
                e_spans.append(curr)
                e_size.append(block)
                e_masks.append(create_entites_mask(*curr, len(self.doc[doc_id].encoding)))
        
        entity_spans = torch.tensor(e_spans, dtype=torch.long)
        entity_masks = torch.stack(e_masks)
        entity_sizes = torch.tensor(e_size, dtype=torch.long)
        entity_sample_mask = torch.tensor([1] * entity_masks.shape[0], dtype=torch.bool)
        
        test_sample = {
            "context_masks":masks,
            "entity_spans":entity_spans,
            "entity_masks":entity_masks,
            "entity_sizes":entity_sizes,
            "entity_sample_masks":entity_sample_mask,
            "encodings":encodings,
        }
        return test_sample
        
    def itr_rel(self, batch_size, order):
        rel_itr = Iterator(batch_size, self.relations, order)
        return rel_itr
        
    def itr_articles(self, batch_size, order):
        art_itr = Iterator(batch_size, self.doc, order)
        return art_itr
    
    def __len__(self):
        return len(self.doc)
        
    
        
        