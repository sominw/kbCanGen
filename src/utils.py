from data_utils import Article

import torch

def get_conembedding(token, h, x):
    
    token_h = h.view(-1, h.shape[-1])
    token_h = token_h[x.contiguous().view(-1) == token, :]
    
    return token_h

def get_batch_index(arr, ix):
    return torch.stack([arr[i][ix[i]] for i in range(ix.shape[0])])

def create_asc_masks(input_ids):
    attn_mask = list()
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attn_mask.append(seq_mask)

def create_entites_mask(begin, end, size):
    mask = torch.zeros(size, dtype=bool)
    mask[begin:end] = 1
    return mask

def create_relations_mask(r0, r1, size):
    begin = None
    end = None
    if r0[1] < r1[0]:
        begin = r0[1]
    else:
        begin = r1[1]
    if r1[0] < r0[1]:
        end = r1[0]
    else:
        end = r0[0]
    mask = torch.zeros(size, dtype=bool)
    mask[begin:end] = 1
    return mask

def assign_trueLabels(arr):
    ents = list()
    rel = list()
    for article in arr:
        ents.append([ent.as_tuple() for ent in article.entities])
        rel.append([rel.as_tuple() for rel in article.relations])
    
    return ents, rel