import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from biencoder_kge import Reranker
from process_data import load_data_split, get_context_representation
import numpy as np
import pickle
import argparse
import torch.nn.functional as F
import logging
from transformers import BertModel, BertTokenizer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset
import json
import pickle

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file, mode='a')        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)

    logger.addHandler(consoleHandler)
    return logger

def calculate_accuracy(max_idxs, correct):
    #acc = 0
    for idx in max_idxs:
        if idx == 0:
           correct+=1
    return correct

def calculate_top1_5(ordered_labels, correct1, correct5):
    for row in ordered_labels:
        if row[0] == 1:
            correct1+=1
        if 1 in row[:5]:
            correct5+=1
    return correct1, correct5
 
def load_dictionary(dictionary_path): 
    ids = []
    names = []
    with open(dictionary_path,'r') as f: 
        for line in f: 
            line = line.strip().split('||') 
            ids.append(line[0])
            names.append(line[1])
    return ids,names


def evaluate(reranker, val_dataloader, criterion, processed_dictionary, neighbors, labels, entity_ids, mesh2cui, cui2idx, device, logger):
    reranker.model.eval()
    total_samples = 0
    eval_correct = 0
    eval_correct_5 = 0
    loss_list = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(val_dataloader,desc="validation")):
                torch.cuda.empty_cache()
                batch = tuple(t.to(device) for t in batch)
                mention_idxs, context_ids = batch    
                candidate_ids = []
                eval_labels = []
                candidate_mesh_ids = []
                for idx in mention_idxs:
                    candidate_ids.append([])
                    eval_labels.append(labels[idx])
                    candidate_mesh_ids.append([])
                    for neighbor in neighbors[idx]:
                        candidate_ids[-1].append(processed_dictionary[neighbor]['ids'])
                        candidate_mesh_ids[-1].append(entity_ids[neighbor])
                
                candidate_ids = torch.tensor(candidate_ids).to(device)
                scores = reranker(context_ids, candidate_ids, candidate_mesh_ids, mesh2cui, cui2idx,device)
                total_samples += scores.shape[0]
                eval_labels = torch.FloatTensor(eval_labels)
                ordered_labels = []
                scores = torch.sigmoid(scores)
                for scores_row, label_row in zip(scores, eval_labels):
                    ordered = np.array([x for _,x in sorted(zip(scores_row, label_row),reverse=True)])
                    ordered_labels.append(ordered)
                
                eval_correct, eval_correct_5 = calculate_top1_5(ordered_labels, eval_correct, eval_correct_5)
                
                loss = criterion(scores, eval_labels.to(device))
                
                loss_list.append(loss.item())

    logger.info("Evaluation completed, with total samples: {}".format(total_samples))           
    return sum(loss_list)/len(loss_list), eval_correct/total_samples, eval_correct_5/total_samples

def load_pickle(path):
    with open(path,'rb') as handle:
        pkl = pickle.load(handle)
    return pkl

def select_field(data, key1, key2=None):
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]
    
def process_mention_data(samples, tokenizer, id_to_idx, debug = False, split = None):
    processed_samples = []
    print("len of id2idx", len(id_to_idx))
    print(debug)
    if debug:
        print("reducing sample size")
        samples = samples[:300]
    not_added = 0
    #mention_idxs = []
    for idx, sample in enumerate(tqdm(samples, desc = "Tokenizing mentions")):
        context = get_context_representation(sample, tokenizer, split)
        label_id = sample['label_id']
        try:
            record = { 
                "mention": sample['mention'],
                "context_tokens": context['tokens'], 
                "context_ids": context['ids'], 
                "mention_idx": idx, 
                "mention_id":sample['mention_id'], 
                "label_title": sample['label'], 
                "label_id":sample['label_id'],       
                "label_idxs": id_to_idx[sample['label_id']]  ### it is a list, cause an entity could have multiple names, so multiple indexes
            }
            processed_samples.append(record)
        except:
            not_added +=1
            print(label_id)
        
    print("not added, due to inconsistency:",not_added)
    context_tensors = torch.tensor(
        select_field(processed_samples, "context_ids"), dtype=torch.long
    )
    print(context_tensors.shape)

    mentiond_idxs = torch.tensor(
        select_field(processed_samples, "mention_idx"), dtype=torch.long
    )
    
    tensor_data = TensorDataset(mentiond_idxs, context_tensors)

    return tensor_data

def main(params):
    
    writer = SummaryWriter()    
    logger = setup_logger('biencoder_logger', './logs/'+params['output_path']+'.log')
    reranker = Reranker(params)
    
    model = reranker.model
    optimizer = reranker.optimizer
    tokenizer = reranker.tokenizer
    device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    #n_gpu = reranker.n_gpu
    train_data = load_data_split("./data/bc5cdr-c/processed/train/train.jsonl")
    val_data = load_data_split("./data/bc5cdr-c/processed/val/val.jsonl")
    test_data = load_data_split("./data/bc5cdr-c/processed/test/test.jsonl")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True, eps=1e-5)
    ####### load id2idx, idx2id

    id2idx_path = "./data/bc5cdr-c/processed/dictionary/id2idx.pkl"
    idx2id_path = "./data/bc5cdr-c/processed/dictionary/idx2id.pkl"
    id2idx = load_pickle(id2idx_path)
    idx2id = load_pickle(idx2id_path)

    # path = "/share/project/biomed/hcd/Masked_EL/data/bc5cdr-c_v1/processed/train/dictionary_embs.pt"
    # path = "/share/project/biomed/hcd/Masked_EL/data/bc5cdr-c_v1/processed/val/id2idx.pkl"
    
    # val_id2idx = load_pickle(path)
    # path = "/share/project/biomed/hcd/Masked_EL/data/bc5cdr-c_v1/processed/val/biobert_dict.pkl"
    # val_entities = load_pickle(path)
    # path = "/share/project/biomed/hcd/Masked_EL/data/bc5cdr-c_v1/processed/val/dictionary_embs.pt"   
     
    train_tensor_data = process_mention_data(train_data, tokenizer, id2idx, params["debug"])  ### tensor_data have context ids and mention ids
    val_tensor_data = process_mention_data(val_data, tokenizer, id2idx, params["debug"])
    batch_size = params['batch_size']
    val_test_batch_size = params["val_test_batch_size"]
    train_dataloader = DataLoader(train_tensor_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_tensor_data, batch_size = val_test_batch_size, shuffle = True)
    

    entity_ids, entity_names = load_dictionary('./data/bc5cdr-c/processed/dictionary/dictionary.txt')
    # dictionary_path = "/data/hcd/work_dir/Masked_EL/data/bc5cdr-c/processed/dictionary/sapbert/dict.pkl"
    # dictionary_entities = load_pickle(dictionary_path)  ## this is the processed dictionary 
    num_train_epochs = params['epoch']

    n_candidates = params['n_candidates']
    
    path = "./data/bc5cdr-c/processed/dictionary/sapbert/train/neighbors.npy"

    train_neighbors = np.load(path)
    path = "./data/bc5cdr-c/processed/dictionary/sapbert/train/neighbor_labels.npy"
    train_neighbor_labels = np.load(path)
    
    
    path = "./data/bc5cdr-c/processed/dictionary/sapbert/val/neighbors.npy"
    val_neighbors = np.load(path)
    path = "./data/bc5cdr-c/processed/dictionary/sapbert/val/neighbor_labels.npy"
    val_neighbor_labels = np.load(path)
    
    best_val = 0
    best_model = None
    output = params['output_path']
    
    model = model.to(device)
    criterion = torch.nn.BCELoss()
    #criterion = torch.nn.CosineEmbeddingLoss()
    kg_encoder = reranker.kg_encoder
    kg_encoder = kg_encoder.to(device)

    processed_entities = load_pickle('./data/bc5cdr-c/processed/dictionary/biobert/dict.pkl')
   
    
    with open('./data/bc5cdr-c/processed/dictionary/mesh2cui.json','r') as f:
        mesh2cui  = json.load(f)
    
    with open('./data/kge/ent2idx.pkl','rb') as handler:
        cui2idx = pickle.load(handler)
    
    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        model.train()
        logger.info("Training")
        torch.cuda.empty_cache()
        gradient_accumulation_steps = 16
        loss_list = []
        correct = 0
        correct_5 = 0
        total_samples = 0
        accumulated_loss = 0
        last_batch_steps = len(train_dataloader) % gradient_accumulation_steps

        for step, batch in enumerate(tqdm(train_dataloader, desc = "Processing minibatches")):
            batch = tuple(t.to(device) for t in batch)
            mention_idxs, context_ids = batch    
            candidate_ids = []
            train_labels = []
            candidate_mesh_ids = []
            for idx in mention_idxs:
                #ground_truth = train_data[idx]['label_id']
                #print(ground_truth)
                candidate_ids.append([])
                train_labels.append(train_neighbor_labels[idx])
                candidate_mesh_ids.append([])
                for neighbor in train_neighbors[idx]:
                    candidate_ids[-1].append(processed_entities[neighbor]['ids'])
                    #print(entity_ids[neighbor])
                    candidate_mesh_ids[-1].append(entity_ids[neighbor]) 
                   
            candidate_ids = torch.tensor(candidate_ids).to(device)
            train_labels = np.array(train_labels)
            train_labels = torch.FloatTensor(train_labels)
            scores = reranker(context_ids, candidate_ids, candidate_mesh_ids, mesh2cui, cui2idx, device) 
            scores = torch.sigmoid(scores)
            total_samples += scores.shape[0]
            #print(torch.isnan(scores).any(), torch.isinf(scores).any())
            

            

            #context_scores = torch.diagonal(probs, dim1=1, dim2=2)
            
            loss =criterion(scores, train_labels.to(device))
            accumulated_loss += loss.item()
            ordered_labels = []
            for scores_row, label_row in zip(scores, train_labels):
                ordered = np.array([x for _,x in sorted(zip(scores_row, label_row),reverse=True)])
                ordered_labels.append(ordered)
            ordered_labels = torch.FloatTensor(ordered_labels)

            correct, correct_5 = calculate_top1_5(ordered_labels, correct, correct_5)
            

            # context, candidate = reranker(context_ids, candidate_ids, candidate_mesh_ids, mesh2cui, cui2idx, device)
            # 
            # loss = criterion(context, candidate, train_labels)
            # loss = loss.reshape(batch_size, n_candidates)

            
            
            loss.backward()
                
            if (step+1) % gradient_accumulation_steps == 0 or (step+1) == len(train_dataloader):
                loss = loss/gradient_accumulation_steps
                
                
                if (step+1) == len(train_dataloader):
                    average_loss = accumulated_loss/last_batch_steps
                average_loss = accumulated_loss / gradient_accumulation_steps
                logger.info("step: {}, loss: {}, accuracy1: {}, accuracy5: {}".format(step, average_loss, correct/total_samples, correct_5/total_samples))
                loss_list.append(average_loss)
                optimizer.step()
                optimizer.zero_grad()
                accumulated_loss = 0
                
        #total_samples = len(train_data)
        logger.info("Training completed, with total samples: {}".format(total_samples))    
        logger.info("Training, epoch: {},  epoch_loss: {}, accuracy1: {}, accuracy5: {}".format(epoch_idx, sum(loss_list)/len(loss_list), correct/total_samples, correct_5/total_samples))
        writer.add_scalar("Trainining loss/epoch 64 candidates", sum(loss_list)/len(loss_list), epoch_idx)
        
        writer.add_scalar("Training accuracy/epoch 64 candidates", correct/total_samples, epoch_idx)
        logger.info("Evaluating") 
        #val_total = len(val_data)
        validation_loss, validation_accuracy, validation_accuracy5 = evaluate(reranker,val_dataloader, criterion, processed_entities,val_neighbors, val_neighbor_labels, entity_ids, mesh2cui, cui2idx, device, logger)
        scheduler.step(validation_loss)
        logger.info("Validation, epoch: {}, loss: {}, accuracy: {}, accuracy5: {}".format(epoch_idx, validation_loss, validation_accuracy, validation_accuracy5))
        
        writer.add_scalar("Validation loss/epoch 64 candidates", validation_loss, epoch_idx)
        writer.add_scalar("Validation accuracy/epoch 64 candidates", validation_accuracy, epoch_idx)
        
        if validation_accuracy > best_val:
            best_val = validation_accuracy
            best_model = {'model': model.state_dict(),
              'optimizer': optimizer.state_dict()}
            if not os.path.exists('./model_ckpts/'+output):
                os.makedirs('./model_ckpts/'+output)
            torch.save(best_model, './model_ckpts/'+output+'/best_model.pt')

        if (epoch_idx+1)%10 == 0:
            checkpoint = {'model': model.state_dict(),
              'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, './model_ckpts/'+output+'/model_ckpt_'+str(epoch_idx)+'.pt')
            
    logger.info("Evaluating on test set")    
    
    path = "./data/bc5cdr-c/processed/dictionary/sapbert/test/neighbors.npy"
    test_neighbors = np.load(path)
    path = "./data/bc5cdr-c/processed/dictionary/sapbert/test/neighbor_labels.npy"
    test_neighbor_labels = np.load(path)
    
    test_tensor_data = process_mention_data(test_data, tokenizer, id2idx, params["debug"])
    test_dataloader = DataLoader(test_tensor_data, batch_size = val_test_batch_size, shuffle = False)
    test_loss, test_accuracy, test_accuracy5 = evaluate(reranker,test_dataloader, criterion, processed_entities, test_neighbors, test_neighbor_labels,entity_ids, mesh2cui, cui2idx, device, logger)
    logger.info("Evaluation loss: {}, Accuracy: {}, Accuracy5: {}".format(test_loss, test_accuracy, test_accuracy5))

    writer.flush()
    writer.close()
       
def eval_test(params):

    print("evaluating")
    reranker = Reranker(params)
    logger = setup_logger('biencoder_logger', './logs/'+params['output_path']+'_biencoder64.log')
    tokenizer = reranker.tokenizer
    device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    
    # path = "./data/bc5cdr-c/processed/dictionary/sapbert/test/neighbors.npy"
    # test_neighbors = np.load(path)
    # path = "/share/project/biomed/hcd/Masked_EL/data/bc5cdr-c_v1/processed/test/neighbor_labels.npy"
    # test_neighbor_labels = np.load(path)

    path = "./data/bc5cdr-c/processed/dictionary/sapbert/test/neighbors.npy"
    test_neighbors = np.load(path)
    path = "./data/bc5cdr-c/processed/dictionary/sapbert/test/neighbor_labels.npy"
    test_neighbor_labels = np.load(path)
    model = reranker.model
    model = model.to(device)

    test_data = load_data_split("./data/bc5cdr-c/processed/test/test.jsonl")

    ####### load id2idx, idx2id

    id2idx_path = "./data/bc5cdr-c/processed/dictionary/id2idx.pkl"
    idx2id_path = "./data/bc5cdr-c/processed/dictionary/idx2id.pkl"
    id2idx = load_pickle(id2idx_path)
    idx2id = load_pickle(idx2id_path)
    criterion = torch.nn.BCELoss()
    kg_encoder = reranker.kg_encoder
    kg_encoder = kg_encoder.to(device)

    processed_entities = load_pickle('./data/bc5cdr-c/processed/dictionary/biobert/dict.pkl')
   
    
    with open('./data/bc5cdr-c/processed/dictionary/mesh2cui.json','r') as f:
        mesh2cui  = json.load(f)
    
    with open('./data/kge/ent2idx.pkl','rb') as handler:
        cui2idx = pickle.load(handler)

    test_tensor_data = process_mention_data(test_data, tokenizer, id2idx, params["debug"])
    test_dataloader = DataLoader(test_tensor_data, batch_size = 128, shuffle = False)
    entity_ids, entity_names = load_dictionary('./data/bc5cdr-c/processed/dictionary/dictionary.txt')
    test_loss, test_accuracy, test_accuracy5 = evaluate(reranker,test_dataloader, criterion, processed_entities, test_neighbors, test_neighbor_labels,entity_ids, mesh2cui, cui2idx, device, logger)
    logger.info("Evaluation loss: {}, Accuracy: {}, Accuracy5: {}".format(test_loss, test_accuracy, test_accuracy5))
    
    
if __name__ == "__main__":
    parameters = {"n_candidates":64, "val_test_batch_size":128, "gradient_accumulation_steps":16, "debug":False, "learning_rate":1e-4, "bert_model":"/data/hcd/work_dir/pretrained_models/biobert-v1.1", "out_dim":768, "epoch":100, "output_path":"bi_kge2", "batch_size":8,  "n_gpu":1, "contrastive":False, "pairwise":False, "data_parallel":False, 'evaluation':False, "ckpt_path":"/data/hcd/work_dir/Masked_EL/model_ckpts/bi_kge/best_model.pt"}
    if not parameters['evaluation']:
        main(parameters)
    else:
        print("evaluating")
        eval_test(parameters)