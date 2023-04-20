import torch
import json
from tqdm import tqdm 
from torch.utils.data import TensorDataset
import faiss
import numpy as np 

ENT_START_TAG = "[unused1]"
ENT_END_TAG = "[unused2]"
#ENT_TITLE_TAG = "[unused3]"

def select_field(data, key1, key2=None):
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]

def load_data_split(data_path):
    dataset = []
    with open(data_path,'r') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return dataset

def load_entities(data_path):
    dictionary = []
    with open(data_path,'r') as f:
        for line in f:
            dictionary.append(json.loads(line.strip()))
    return dictionary
    
def get_context_representation(sample, tokenizer, max_length = 512):
    
    mention = sample['mention'].lower()
    context_left = sample['context_left']
    context_right = sample['context_right']
    mention_tokens = tokenizer.tokenize(mention)
    masked_mention = [ENT_START_TAG] + ["[MASK]"] * len(mention_tokens) + [ENT_END_TAG]
    context_left = tokenizer.tokenize(context_left)
    context_right = tokenizer.tokenize(context_right)
    
    left_quota = (max_length - len(masked_mention)) // 2 - 1
    right_quota = max_length - len(masked_mention) - left_quota - 2
    left_add = len(context_left)
    right_add = len(context_right)

    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if len(context_right) <= right_quota:
            left_quota += right_quota - right_add
            
    context_tokens = (
        context_left[-left_quota:] + masked_mention + context_right[:right_quota]
    )
     
    context_tokens = ["[CLS]"] + context_tokens + ["[SEP]"]
    
    input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
    padding = [0] * (max_length - len(input_ids))
    input_ids += padding
    
    return {
        "tokens": context_tokens,
        "ids": input_ids,
    }
    


def get_candidate_representation(sample, tokenizer, max_length = 25):  ### from entity_dictionary
    candidate_tokens = tokenizer.tokenize(sample.lower())
    if len(candidate_tokens)>max_length:
        candidate_tokens = candidate_tokens[:max_length]
        
    input_ids = tokenizer.convert_tokens_to_ids(candidate_tokens)
    padding = [0] * (max_length - len(input_ids))
    input_ids += padding
    
    return {
        "tokens": candidate_tokens,
        "ids": input_ids
    }

def process_entity_dictionary(entities, tokenizer, dictionary_processed = False):
    id_to_idx = {}
    entity_dictionary = []
    for idx, ent in enumerate(tqdm(entities, desc="Tokenizing dictionary")):
        id_to_idx[ent['mention_id']] = idx
        if not dictionary_processed:
            label_representation = get_candidate_representation(ent["mention"], tokenizer)
            entity_dictionary.append({
                "tokens": label_representation["tokens"],
                "ids": label_representation["ids"]
            }) 
            
            # entity_dictionary[idx]["tokens"] = label_representation["tokens"]
            # entity_dictionary[idx]["ids"] = label_representation["ids"]

    return id_to_idx, entity_dictionary

def process_mention_data(samples, tokenizer, id_to_idx, debug = False, max_length=512):
    processed_samples = []
    if debug:
        print("reducing sample size")
        samples = samples[:100]
    for idx, sample in enumerate(tqdm(samples, desc = "Tokenizing mentions")):
        context = get_context_representation(sample, tokenizer)
        label_id = sample['label_id']
        record = {
            "mention": sample['mention'],
            "context_tokens": context['tokens'],
            "context_ids": context['ids'],
            "mention_idx": idx,
            "label_title": sample['label_title'],
            "label_id":sample['label_id'],
            "label_idxs": id_to_idx[sample['label_id']]
            
        }
        processed_samples.append(record)
        context_tensors = torch.tensor(
            select_field(processed_samples, "context_ids"), dtype=torch.long
        )

        label_idxs = torch.tensor(
        select_field(processed_samples, "label_idxs"), dtype=torch.long,
        )

        tensor_data = TensorDataset(context_tensors, label_idxs)

    return processed_samples, tensor_data

def process_candidate_data(model, device,  entity_dictionary,debug=False, max_length=50):
    ent_embs = [] 
    
    with torch.no_grad():
        for idx, ent in enumerate(tqdm(entity_dictionary, desc = "calculating embeddings")):
            input = torch.tensor(ent['ids'])
            #print(input.shape)
            _, emb = model(candidate_ids = input[None,:].to(device) )
            
            ent_embs.append(emb.cpu())

    ent_embs = torch.stack(ent_embs)
    print(ent_embs.shape)
    d = 768
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(d)
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.add(ent_embs)
    top_k = 65
    neighbor_list = []
    D, I = gpu_index_flat.search(ent_embs, top_k)
    
    for idx, row  in enumerate(I):
        if row[0] == idx:
            #print("it's the same")
            neighbor_list.append(row[:-1])
        else:
            #print("not the same")
            to_append = []
            to_append.append(idx)
            
            for elem in row:
                if elem != idx:
                    to_append.append(elem)
                if len(to_append)  == top_k-1 :
                    break
            neighbor_list.append(to_append)
    
    return ent_embs, neighbor_list

