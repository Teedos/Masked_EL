import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import numpy as np
from biencoder import BiEncoderRanker
from process_data import load_data_split, load_entities, process_entity_dictionary, process_mention_data, process_candidate_data
import pickle
import argparse
import torch.nn.functional as F

def main(params):
    model_output_path = params['output_path']
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

    reranker = BiEncoderRanker(params)
    model = reranker.model
    optimizer = reranker.optimizer
    tokenizer = reranker.tokenizer
    device = reranker.device
    n_gpu = reranker.n_gpu

    train_data = load_data_split("./data/bc5cdr-c/processed/train.jsonl")
    val_data = load_data_split("./data/bc5cdr-c/processed/val.jsonl")
    test_data = load_data_split("./data/bc5cdr-c/processed/test.jsonl")

    entities = load_entities("./data/bc5cdr-c/entity_documents.json")

    id_to_idx, entity_dictionary = process_entity_dictionary(entities, tokenizer, dictionary_processed = False)
    
    processed_train_mentions, train_tensor_data = process_mention_data(train_data, tokenizer, id_to_idx, params["debug"])
    #processed_val_mentions, val_tensor_data = process_mention_data(val_data, tokenizer, id_to_idx)
    #processed_test_mentions, test_tensor_data = process_mention_data(test_data, tokenizer, id_to_idx)
    batch_size = params['batch_size']
    train_dataloader = DataLoader(train_tensor_data, batch_size=batch_size, shuffle=True)
    #val_dataloader = DataLoader(val_tensor_data, batch_size = batch_size, shuffle = False)

    num_train_epochs = params['epoch']

    ent_embs, neighbor_list = process_candidate_data(model, entity_dictionary, params["debug"])

    entity_dictionary_pkl_path = './processed/entity_dictionary.pkl'
    
    with open(entity_dictionary_pkl_path, 'wb') as write_handle:
                    pickle.dump(entity_dictionary, write_handle,
                                protocol=pickle.HIGHEST_PROTOCOL)

    entity_embedding_pkl_path = './processed/entity_embedding.pkl'


    with open(entity_embedding_pkl_path, 'wb') as write_handle:
                    pickle.dump(ent_embs, write_handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
    
    np.save('./processed/neighbor_list.npy', neighbor_list)
    model.cuda()
    
    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        model.train()
        torch.cuda.empty_cache()
        tr_loss = 0
        #print("epoch: ", epoch_idx)
        for step, batch in enumerate(train_dataloader):
            
            batch = tuple(t.to(device) for t in batch)
            context_ids, label_idxs = batch    
               
            #### Need to add negative samples, can take the top 64 most similar to the candidate
            #### Target needs to be [1, 0, 0, 0, 0, 0, 0, 0]  target needs to be of length = 64
            #print("shape of ids: ", context_ids.shape)
            candidate_ids = []
            #print(len(label_idxs))
            for idx in label_idxs:
                candidate_ids.append([])
                # if neighbor_list[idx][0] != idx:
                #     print("something is wrong") 
                # else:
                #     print("correct")
                for neighbor in neighbor_list[idx]:
                    #print(neighbor)
                    
                    candidate_ids[-1].append(entity_dictionary[neighbor]['ids'])
            
            candidate_ids = torch.tensor(candidate_ids)
            #print("candidates shape",candidate_ids.shape)

            target = [1] + [0] * (len(label_idxs)-1)
            #print(len(target))
            target = torch.tensor(target).unsqueeze(1)
            
            context_embs, candidate_embs , scores = reranker(context_ids, candidate_ids, target) 

            loss = F.cross_entropy(scores, target, reduction="mean") 
            #print(loss.item())
            tr_loss += loss.item()
        batch_loss = tr_loss/len(train_dataloader)
        print("epoch:{}, average loss:{}".format(epoch_idx, batch_loss))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # Input data
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--epoch",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--out_dim",
        default=768,
        type=int,
        required=False,
    )

    parser.add_argument(
        "--n_gpu",
        default=1,
        type=int,
        required=False,
    )

    parser.add_argument(
        "--bert_model",
        default="/home/DAIR/hongcd/DAIR_Biomed/EL/arboEL/models/biobert-v1.1",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=int,
        required=False,
    )

    parser.add_argument(
        "--debug",
        type=bool,
        required=False,
    )

    args = parser.parse_args()

    parameters = args.__dict__
    main(parameters)