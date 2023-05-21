from tqdm import tqdm
from datasets import PretrainDataset
import torch


def select_field(data, key1, key2=None):
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]

# def select_field(data, key1, key2=None):
#     if key2 is None:
#         return [torch.tensor(example[key1],dtype=torch.long) for example in data]
#     else:
#         return [torch.tensor(example[key1][key2],dtype = torch.long) for example in data]    
    
def process_data(cuis,names, definitions, tokenizer, candidate_max_length = 30, context_max_length=512 ):
    processed_samples = []
    
    for idx, (cui, definition, name) in tqdm(enumerate(zip(cuis, definitions, names)), total=len(definitions),desc= "Processing Samples"):
        definition_tokens = tokenizer.tokenize(definition)
        encoded_definition = tokenizer.encode_plus(definition_tokens, add_special_tokens=True, padding='max_length', truncation=True, max_length = context_max_length)
        definition_token_ids = encoded_definition['input_ids']
        definition_attn_mask = encoded_definition['attention_mask']
        # if len(definition_tokens) > context_max_length - 2:
        #     definition_tokens = definition_tokens[:context_max_length - 2]
        #     definition_tokens = [tokenizer.cls_token] + definition_tokens + [tokenizer.sep_token]
        # definition_token_ids = tokenizer.convert_tokens_to_ids(definition_tokens)
        # definition_padding = [0] * (context_max_length - len(definition_token_ids))
        # definition_token_ids +=definition_padding
        
        name_tokens = tokenizer.tokenize(name)
        encoded_name = tokenizer.encode_plus(name_tokens, padding='max_length', truncation=True, max_length=candidate_max_length)
        name_token_ids = encoded_name['input_ids']
        name_attn_mask = encoded_name['attention_mask']
        # name_tokens = name_tokens[:candidate_max_length]
        
        # name_token_ids = tokenizer.convert_tokens_to_ids(name_tokens)
        
        # name_padding =  [0] * (candidate_max_length - len(name_token_ids))
        # name_token_ids += name_padding
        record = {
            "index_in_file": idx,
            "definition_token_ids": definition_token_ids,
            "definition_attention_mask":definition_attn_mask,
            "name_token_ids": name_token_ids,
            "name_attention_mask": name_attn_mask,
            "cui": cui
        }
        processed_samples.append(record)
    
    # print("Definition Lengths:", [len(sample["definition_token_ids"]) for sample in processed_samples])
    # print("Name Lengths:", [len(sample["name_token_ids"]) for sample in processed_samples])
    # print("Indexes Lengths:", [len(sample["index_in_file"]) for sample in processed_samples])
    indexes = torch.tensor(select_field(processed_samples, "index_in_file"),dtype=torch.long)
    #print(type(indexes))
    definition_ids = torch.tensor(select_field(processed_samples, "definition_token_ids"),dtype=torch.long)

    definition_masks =  torch.tensor(select_field(processed_samples, "definition_attention_mask"),dtype=torch.long)
    #print(select_field(processed_samples, "name_token_ids"))
    name_ids = torch.tensor(select_field(processed_samples, "name_token_ids"),dtype=torch.long)

    name_masks =  torch.tensor(select_field(processed_samples, "name_attention_mask"),dtype=torch.long)
    #print(definition_ids.shape, name_ids.shape)
    #print(indexes.shape, definition_ids.shape, name_ids.shape)
    tensor_data = PretrainDataset(name_ids, name_masks, definition_ids, definition_masks, indexes)

    return processed_samples, tensor_data