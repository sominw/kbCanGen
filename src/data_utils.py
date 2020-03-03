"""
<x>id<x>: identifier within the dataset
index: identifier in the document
"""

class Iterator:
    def __init__(self, batch_size, content, order = None):
        self.batch_size = batch_size
        self.content = content
        self.length = len(self.content)
        self.ix = 0
        if (order is None):
            self.order = list(range(len(self.content)))
        else:
            self.order = order
            
    def __iter__(self):
        return self
    
    def __next__(self):
        temp = list()
        for i in self.order[self.ix:self.ix + self.batch_size]:
            temp.append(self.content[i])
        self.ix += self.batch_size
        return temp

class Token:
    def __init__(self, start, end, idn, index, string):
        self.index = index
        self.idn = idn
        self.start = start
        self.end = end
        self.string = string
        
    def astuple(self):
        return (self.start, self.end)
        
    def __repr__(self):
        return self.string
    
    def __hash__(self):
        return hash(self.idn)
    
class Span:
    def __init__(self, tokens):
        self.tokens = tokens
        self.start = tokens[0].start
        self.end = tokens[-1].end
        
    def astuple(self):
        return (self.start, self.end)
        
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, i):
        return self.tokens[i]
    
    def __iter__(self):
        return iter(self.tokens)

class EntityType:
    def __init__(self, name, index, eid):
        self.eid = eid
        self.index = index
        self.name = name
        
    def __hash__(self):
        return hash(self.eid)
    
    def __int__(self):
        return self.index

class Entity:
    def __init__(self, e_type, tokens, string, eid):
        self.e_type = e_type
        self.tokens = tokens
        self.string = string
        self.eid = eid
        self.span_start = self.tokens[0].start
        self.span_end = self.tokens[-1].end
        self.span = (self.span_start, self.span_end)
        self.string = self.string
        
    def astuple(self):
        return (self.span_start, self.span_end, self.e_type)

    def __len__(self):
        return len(self.tokens)
    
    def __str__(self):
        return self.string
        
    def __hash__(self):
        return hash(self.eid)

class RelationType:
    def __init__(self, rel_type, index, rel_id):
        self.rel_id = rel_id
        self.index = index
        self.rel_type = rel_type
        
    def __int__(self):
        return self.index
    
    def __hash__(self):
        return self.rel_id
    
class Relation:
    def __init__(self, rel_type, rel_id, first_e, second_e):
        self.rel_id = rel_id
        self.rel_type = rel_type
        self.first_entity = first_e
        self.second_entity = second_e
        
    def astuple(self):
        return ((self.first_entity.span_start, self.first_entity.span_end, self.first_entity.e_type),
                (self.second_entity.span_start, self.second_entity.span_end, self.second_entity.e_type),
                self.rel_type)
    
    def __hash__(self):
        return hash(self.rel_id)
    
class Article:
    def __init__(self, pmid, encoding, entities, relations, tokens):
        self.pmid = pmid
        self.entities = entities
        self.relations = relations
        self._tokens = tokens
        self.encoding = encoding
    
    @property
    def tokens(self):
        return Span(self._tokens)
    
    def __hash__(self):
        return hash(self.pmid)
    
        
