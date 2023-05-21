import faiss
from process_data import get_candidate_representation
from tqdm import tqdm
import torch
import numpy as np 
from transformers import BertTokenizer, BertModel
import pickle
import json

def get_candidates(mention_samples, model, tokenizer, device, entity_embs, id2idx, idx2id, n_candidates, debug=False):
    #id2emb = {}
    d = 768
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(d)
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.add(entity_embs.cpu())
    count = 0
    targets = []
    neighbor_list = []
    rec1 = 0
    for idx, sample in enumerate(tqdm(mention_samples, desc = "calculating embeddings")):
            with torch.no_grad():
                mention_representation = get_candidate_representation(sample['mention'].lower(), tokenizer)
                mention_ids = mention_representation['ids']
                input = torch.tensor(mention_ids)
                emb = model(input[None,:].to(device))[0].mean(1)

            mention_label = sample['label_id']

            emb = emb.cpu()
            #id2emb[sample['mention_id']] = emb
            
            D, I = gpu_index_flat.search(emb, n_candidates)
            result = I[0]
            neighbor_ids = [idx2id[k] for k in result]
            if mention_label in neighbor_ids:
                count +=1
            target = []

            for id_ in neighbor_ids:
                if id_ == mention_label:
                    target.append(1)
                else:
                    target.append(0)
            
            targets.append(target[0:n_candidates])
            neighbor_list.append(result[0:n_candidates])
       
    print(count/len(mention_samples))
    return neighbor_list, targets

def load_data_split(data_path):
    dataset = []
    with open(data_path,'r') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return dataset

def load_pickle(path):
    with open(path,'rb') as f:
        my_dict = pickle.load(f)
    return my_dict

def main():

    ####global data
    train_data_path = "./data/bc5cdr-c/processed/train/train.jsonl"
    train_data = load_data_split(train_data_path)

    val_data_path = "./data/bc5cdr-c/processed/val/val.jsonl"
    val_data = load_data_split((val_data_path))

    test_data_path = "./data/bc5cdr-c/processed/test/test.jsonl"
    test_data = load_data_split(test_data_path)

    device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    
    n_candidates = 64

    id2idx_path = "./data/bc5cdr-c/processed/dictionary/id2idx.pkl"
    idx2id_path = "./data/bc5cdr-c/processed/dictionary/idx2id.pkl"
    id2idx = load_pickle(id2idx_path)
    idx2id = load_pickle(idx2id_path)
    
    ### biobert

 
    # ent_embs_path = './data/bc5cdr-c/embeddings/biobert_dictionary_embs.pt'
    # entity_embs = torch.load(ent_embs_path)
    # model_path = "/data/hcd/work_dir/pretrained_models/biobert-v1.1"
    # model = BertModel.from_pretrained(model_path)
    # tokenizer = BertTokenizer.from_pretrained(model_path)
    # model.to(device)

    # train_neighbors, train_labels = get_candidates(train_data, model, tokenizer, device, entity_embs, id2idx, idx2id, n_candidates)
    # np.save('./data/bc5cdr-c/processed/dictionary/biobert/train/neighbors.npy', train_neighbors)
    # np.save('./data/bc5cdr-c/processed/dictionary/biobert/train/neighbor_labels.npy', train_labels) 

    # val_neighbors, val_labels = get_candidates(val_data, model, tokenizer, device, entity_embs, id2idx, idx2id, n_candidates)
    # np.save('./data/bc5cdr-c/processed/dictionary/biobert/val/neighbors.npy', val_neighbors)
    # np.save('./data/bc5cdr-c/processed/dictionary/biobert/val/neighbor_labels.npy', val_labels)

    # test_neighbors, test_labels = get_candidates(test_data, model, tokenizer, device, entity_embs, id2idx, idx2id, n_candidates)
    # np.save('./data/bc5cdr-c/processed/dictionary/biobert/test/neighbors.npy', test_neighbors)
    # np.save('./data/bc5cdr-c/processed/dictionary/biobert/test/neighbor_labels.npy', test_labels)

    ### sapbert

    
    # ent_embs_path = './data/bc5cdr-c/embeddings/sapbert_dictionary_embs.pt'
    # entity_embs = torch.load(ent_embs_path)
    # model_path = "/data/hcd/work_dir/pretrained_models/SapBERT"
    # model = BertModel.from_pretrained(model_path)
    # tokenizer = BertTokenizer.from_pretrained(model_path)
    # model.to(device)

    # train_neighbors, train_labels = get_candidates(train_data, model, tokenizer, device, entity_embs, id2idx, idx2id, n_candidates)
    # np.save('./data/bc5cdr-c/processed/dictionary/sapbert/train/neighbors.npy', train_neighbors)
    # np.save('./data/bc5cdr-c/processed/dictionary/sapbert/train/neighbor_labels.npy', train_labels) 

    # val_neighbors, val_labels = get_candidates(val_data, model, tokenizer, device, entity_embs, id2idx, idx2id, n_candidates)
    # np.save('./data/bc5cdr-c/processed/dictionary/sapbert/val/neighbors.npy', val_neighbors)
    # np.save('./data/bc5cdr-c/processed/dictionary/sapbert/val/neighbor_labels.npy', val_labels)

    # test_neighbors, test_labels = get_candidates(test_data, model, tokenizer, device, entity_embs, id2idx, idx2id, n_candidates)
    # np.save('./data/bc5cdr-c/processed/dictionary/sapbert/test/neighbors.npy', test_neighbors)
    # np.save('./data/bc5cdr-c/processed/dictionary/sapbert/test/neighbor_labels.npy', test_labels)

if __name__ == "__main__":
    main()