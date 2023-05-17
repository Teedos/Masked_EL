import torch
import json
from tqdm import tqdm 
from torch.utils.data import TensorDataset
import faiss
import numpy as np 
from transformers import BertTokenizer, BertModel
import pickle
from process_data import load_data_split, get_candidate_representation

def load_dictionary(dictionary_path): 
    ids = []
    names = []
    with open(dictionary_path,'r') as f: 
        for line in f: 
            line = line.strip().split('||') 
            ids.append(line[0])
            names.append(line[1])
    return ids,names


biosyn_path = "/share/project/biomed/hcd/BioSyn/pretrained/biosyn-sapbert-bc5cdr-chemical"
train_data = load_data_split("./data/bc5cdr-c_v1/processed/train.jsonl")
val_data = load_data_split("./data/bc5cdr-c_v1/processed/val.jsonl")
test_data = load_data_split("./data/bc5cdr-c_v1/processed/test.jsonl")
#entities = load_entities("./data/bc5cdr-c_v1/entity_documents.json")
path = "/share/project/biomed/hcd/BioSyn/datasets/bc5cdr-chemical/train_dictionary.txt"
train_dict_ids, train_dict_names = load_dictionary(path)
# train_dict_ids = train_dict_ids[:100]
# train_dict_names = train_dict_names[:100]
path = "/share/project/biomed/hcd/BioSyn/datasets/bc5cdr-chemical/dev_dictionary.txt"
val_dict_ids, val_dict_names = load_dictionary(path)
# val_dict_ids = val_dict_ids[:100]
# val_dict_names = val_dict_names[:100]
path = "/share/project/biomed/hcd/BioSyn/datasets/bc5cdr-chemical/test_dictionary.txt"
test_dict_ids, test_dict_names = load_dictionary(path)
# test_dict_ids = test_dict_ids[:100]
# test_dict_names = test_dict_names[:100]
#biosyn = BertModel.from_pretrained(biosyn_path)

biosyn_tokenizer = BertTokenizer.from_pretrained(biosyn_path)

biobert_tokenizer = BertTokenizer.from_pretrained("/share/project/biomed/hcd/arboEL/models/biobert-base-cased-v1.1")
device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

#biosyn = biosyn.to(device)


def process_entity_dictionary(ids, names, tokenizer, biobert_tokenizer, dictionary_processed = False):
    ## tokenizer has to be biosyn tokenizer
    idx_to_id = {}
    id_to_idx = {}
    entity_dictionary = []
    with torch.no_grad():
        for idx, (id_, name) in enumerate(zip(ids,names)):
            print(idx,end='\r')
            #id_to_idx[id_] = idx
            idx_to_id[idx] = id_
            if id_ not in id_to_idx:
                id_to_idx[id_] = []
            id_to_idx[id_].append(idx)
            if not dictionary_processed:
                label_representation = get_candidate_representation(name.lower(), tokenizer)
                biobert_representation = get_candidate_representation(name.lower(),biobert_tokenizer)
                entity_dictionary.append({
                    "tokens": label_representation["tokens"],
                    "ids": label_representation["ids"],
                    "biobert_tokens": biobert_representation["tokens"],
                    "biobert_ids": biobert_representation["ids"]
                }) 
                
                
    return id_to_idx, idx_to_id, entity_dictionary

train_id2idx, train_idx2id, train_dictionary = process_entity_dictionary(train_dict_ids, train_dict_names, biosyn_tokenizer, biobert_tokenizer)
# with open("./data/bc5cdr-c_v1/processed/train/idx2id.pkl",'wb') as handle:
#     pickle.dump(train_idx2id, handle)
    
# with open("./data/bc5cdr-c_v1/processed/train/id2idx.pkl",'wb') as handle:
#     pickle.dump(train_id2idx, handle)

with open("./data/bc5cdr-c_v1/processed/train/biobert_dict.pkl", 'wb') as write_handle:
                    pickle.dump(train_dictionary, write_handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
val_id2idx, val_idx2id, val_dictionary = process_entity_dictionary(val_dict_ids, val_dict_names, biosyn_tokenizer, biobert_tokenizer)
# with open("./data/bc5cdr-c_v1/processed/val/idx2id.pkl",'wb') as handle:
#     pickle.dump(val_idx2id, handle)
# with open("./data/bc5cdr-c_v1/processed/val/id2idx.pkl",'wb') as handle:
#     pickle.dump(val_id2idx, handle)

with open("./data/bc5cdr-c_v1/processed/val/biobert_dict.pkl", 'wb') as write_handle:
                    pickle.dump(val_dictionary, write_handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
                    
test_id2idx, test_idx2id, test_dictionary = process_entity_dictionary(test_dict_ids, test_dict_names, biosyn_tokenizer, biobert_tokenizer)
# with open("./data/bc5cdr-c_v1/processed/test/idx2id.pkl",'wb') as handle:
#     pickle.dump(test_idx2id, handle)
# with open("./data/bc5cdr-c_v1/processed/test/id2idx.pkl",'wb') as handle:
#     pickle.dump(test_id2idx, handle)
with open("./data/bc5cdr-c_v1/processed/test/biobert_dict.pkl", 'wb') as write_handle:
                    pickle.dump(test_dictionary, write_handle,
                                protocol=pickle.HIGHEST_PROTOCOL)