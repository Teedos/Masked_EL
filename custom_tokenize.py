import torch
import json
from tqdm import tqdm 
from torch.utils.data import TensorDataset
import faiss
import numpy as np 
from transformers import BertTokenizer, BertModel
import pickle
from process_data import load_data_split, get_candidate_representation

def embed_dictionary(model, device ,entity_dictionary):
    ent_embs = [] 
    with torch.no_grad():
        for idx, ent in enumerate(tqdm(entity_dictionary, desc = "calculating embeddings")):
            input = torch.tensor(ent['ids'])
            #print(input.shape)
            emb = model(input[None,:].to(device))[0].mean(1).squeeze(0)
            #print(emb.shape)
            ent_embs.append(emb)
    ent_embs = torch.stack(ent_embs)
    return ent_embs

def load_dictionary(dictionary_path): 
    ids = []
    names = []
    with open(dictionary_path,'r') as f: 
        for line in f: 
            line = line.strip().split('||') 
            meshs = line[0].split('|')
            for mesh in meshs:
                ids.append(mesh)  
                names.append(line[1])
    return ids,names


sapbert_path = "/data/hcd/work_dir/pretrained_models/SapBERT"
biobert_path = "/data/hcd/work_dir/pretrained_models/biobert-v1.1"
# train_data = load_data_split("./data/bc5cdr-c_v1/processed/train.jsonl")
# val_data = load_data_split("./data/bc5cdr-c_v1/processed/val.jsonl")
# test_data = load_data_split("./data/bc5cdr-c_v1/processed/test.jsonl")
#entities = load_entities("./data/bc5cdr-c_v1/entity_documents.json")
path = "/data/hcd/work_dir/Masked_EL/data/bc5cdr-c/processed/dictionary/dictionary.txt"
dict_ids, dict_names = load_dictionary(path)
# train_dict_ids = train_dict_ids[:100]
# train_dict_names = train_dict_names[:100]
# path = "/share/project/biomed/hcd/BioSyn/datasets/bc5cdr-chemical/dev_dictionary.txt"
# val_dict_ids, val_dict_names = load_dictionary(path)
# # val_dict_ids = val_dict_ids[:100]
# # val_dict_names = val_dict_names[:100]
# path = "/share/project/biomed/hcd/BioSyn/datasets/bc5cdr-chemical/test_dictionary.txt"
# test_dict_ids, test_dict_names = load_dictionary(path)
# test_dict_ids = test_dict_ids[:100]
# test_dict_names = test_dict_names[:100]
#biosyn = BertModel.from_pretrained(biosyn_path)

sapbert_tokenizer = BertTokenizer.from_pretrained(sapbert_path)
biobert_tokenizer = BertTokenizer.from_pretrained(biobert_path)
device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

def process_entity_dictionary(ids, names, tokenizer, dictionary_processed = False):
    
    idx_to_id = {}
    id_to_idx = {}
    entity_dictionary = []
    with torch.no_grad():
        for idx, (id_, name) in enumerate(zip(ids,names)):
            print(idx,end='\r')
            idx_to_id[idx] = id_
            if id_ not in id_to_idx:
                id_to_idx[id_] = []
            id_to_idx[id_].append(idx)
            if not dictionary_processed:
                label_representation = get_candidate_representation(name.lower(), tokenizer)
                entity_dictionary.append({
                    "tokens": label_representation["tokens"],
                    "ids": label_representation["ids"],
                }) 
                
                
    return id_to_idx, idx_to_id, entity_dictionary

biobert_id2idx, biobert_idx2id, biobert_dictionary = process_entity_dictionary(dict_ids, dict_names, biobert_tokenizer)

with open("./data/bc5cdr-c/processed/dictionary/biobert/dict.pkl", 'wb') as write_handle:
                    pickle.dump(biobert_dictionary, write_handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
# with open("./data/bc5cdr-c/processed/dictionary/biobert/idx2id.pkl",'wb') as handle:
#     pickle.dump(biobert_idx2id, handle)
# with open("./data/bc5cdr-c/processed/dictionary/biobert/id2idx.pkl",'wb') as handle:
#     pickle.dump(biobert_id2idx, handle)

# biobert_model = BertModel.from_pretrained(biobert_path)
# biobert_model.to(device)
# biobert_ent_embs = embed_dictionary(biobert_model, device, biobert_dictionary)
# torch.save(biobert_ent_embs, './data/bc5cdr-c/embeddings/biobert_dictionary_embs.pt')

#sapbert_id2idx, sapbert_idx2id, sapbert_dictionary = process_entity_dictionary(dict_ids, dict_names, sapbert_tokenizer)

# with open("./data/bc5cdr-c/processed/dictionary/sapbert/dict.pkl", 'wb') as write_handle:
#                     pickle.dump(sapbert_dictionary, write_handle,
#                                 protocol=pickle.HIGHEST_PROTOCOL)
# with open("./data/bc5cdr-c/processed/dictionary/idx2id.pkl",'wb') as handle:
#     pickle.dump(sapbert_idx2id, handle)
# with open("./data/bc5cdr-c/processed/dictionary/id2idx.pkl",'wb') as handle:
#     pickle.dump(sapbert_id2idx, handle)

# with open("./data/bc5cdr-c/processed/dictionary/biobert/dict.pkl", 'rb') as handle:
#     biobert_dictionary = pickle.load(handle)

# with open("./data/bc5cdr-c/processed/dictionary/sapbert/dict.pkl", 'rb') as handle:
#     sapbert_dictionary = pickle.load(handle)




# sapbert_model = BertModel.from_pretrained(sapbert_path)
# sapbert_model.to(device)
# sapbert_ent_embs = embed_dictionary(sapbert_model, device, sapbert_dictionary)
# torch.save(sapbert_ent_embs, './data/bc5cdr-c/embeddings/sapbert_dictionary_embs.pt')
