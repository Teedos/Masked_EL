import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from biencoder import BiEncoderRanker
from process_data import load_data_split, get_context_representation
import numpy as np
import pickle
import argparse
import torch.nn.functional as F
import logging
from transformers import BertModel, BertTokenizer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset

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



def evaluate(reranker, val_dataloader, criterion, entities, neighbors, labels, device, logger):
    reranker.model.eval()
    total_samples = 0
    eval_correct = 0
    eval_correct_5 = 0
    loss_list = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(val_dataloader,desc="validation mini batches")):
                torch.cuda.empty_cache()
            
                batch = tuple(t.to(device) for t in batch)
                mention_idxs, context_ids = batch    
                candidate_ids = []
                eval_labels = []
                for idx in mention_idxs:
                    candidate_ids.append([])
                    eval_labels.append(labels[idx])
                    for neighbor in neighbors[idx]:
                        candidate_ids[-1].append(entities[neighbor]['biobert_ids'])
                
                candidate_ids = torch.tensor(candidate_ids).to(device)
                #print(candidate_ids.shape)
                scores = reranker(context_ids, candidate_ids, device) 
                total_samples += scores.shape[0]
                
                eval_labels = torch.FloatTensor(eval_labels)
                # max_idxs = scores.argmax(dim=1)
                # correct = calculate_accuracy(max_idxs, correct)
                ordered_labels = []
                for scores_row, label_row in zip(scores, eval_labels):
                    ordered = np.array([x for _,x in sorted(zip(scores_row, label_row),reverse=True)])
                    ordered_labels.append(ordered)
                #ordered_labels = 
                #ordered_labels = torch.FloatTensor(ordered_labels)
                eval_correct, eval_correct_5 = calculate_top1_5(ordered_labels, eval_correct, eval_correct_5)
                probs = F.softmax(scores, dim=1)

                #p_correct = probs[:, 0]
                
                loss = criterion(probs, eval_labels.to(device))
                
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
    
def process_mention_data(samples, tokenizer, id_to_idx, split = None, debug = False, max_length=512):
    processed_samples = []
    print("len of id2idx", len(id_to_idx))
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
                #"label_title": sample['label_title'],
                "label_title": sample['label'],
                "label_id":sample['label_id'],
                
                "label_idxs": id_to_idx[sample['label_id']]
                
            }
            processed_samples.append(record)
        except:
            not_added +=1
        
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
    logger = setup_logger('biencoder_logger', './logs/'+params['output_path']+'_biencoder64.log')
    reranker = BiEncoderRanker(params)
    
    model = reranker.model
    optimizer = reranker.optimizer
    tokenizer = reranker.tokenizer
    device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    #reranker = reranker.to(device)
    #model = model.to(device)
    n_gpu = reranker.n_gpu
    #biosyn_path = "/share/project/biomed/hcd/BioSyn/pretrained/biosyn-sapbert-bc5cdr-chemical"
    train_data = load_data_split("./data/bc5cdr-c_v1/processed/train.jsonl")
    val_data = load_data_split("./data/bc5cdr-c_v1/processed/val.jsonl")
    test_data = load_data_split("./data/bc5cdr-c_v1/processed/test.jsonl")

    path = "/share/project/biomed/hcd/Masked_EL/data/bc5cdr-c_v1/processed/train/id2idx.pkl"
    train_id2idx = load_pickle(path)
    #print(train_id2idx['D002216'])
    path = "/share/project/biomed/hcd/Masked_EL/data/bc5cdr-c_v1/processed/train/biobert_dict.pkl"
    train_entities = load_pickle(path)

    path = "/share/project/biomed/hcd/Masked_EL/data/bc5cdr-c_v1/processed/train/dictionary_embs.pt"
    train_ent_embs = torch.load(path)
    path = "/share/project/biomed/hcd/Masked_EL/data/bc5cdr-c_v1/processed/val/id2idx.pkl"
    
    val_id2idx = load_pickle(path)
    path = "/share/project/biomed/hcd/Masked_EL/data/bc5cdr-c_v1/processed/val/biobert_dict.pkl"
    val_entities = load_pickle(path)
    path = "/share/project/biomed/hcd/Masked_EL/data/bc5cdr-c_v1/processed/val/dictionary_embs.pt"
    val_ent_embs = torch.load(path)
    
    train_tensor_data = process_mention_data(train_data, tokenizer, train_id2idx, params["debug"])
    val_tensor_data = process_mention_data(val_data, tokenizer, val_id2idx, params["debug"])
    batch_size = params['batch_size']
    val_test_batch_size = params["val_test_batch_size"]
    train_dataloader = DataLoader(train_tensor_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_tensor_data, batch_size = val_test_batch_size, shuffle = True)

    num_train_epochs = params['epoch']

    n_candidates = params['n_candidates']
    
    path = "/share/project/biomed/hcd/Masked_EL/data/bc5cdr-c_v1/processed/train/neighbors.npy"

    train_neighbors = np.load(path)
    path = "/share/project/biomed/hcd/Masked_EL/data/bc5cdr-c_v1/processed/train/neighbor_labels.npy"
    train_neighbor_labels = np.load(path)
    
    
    path = "/share/project/biomed/hcd/Masked_EL/data/bc5cdr-c_v1/processed/val/neighbors.npy"
    val_neighbors = np.load(path)
    path = "/share/project/biomed/hcd/Masked_EL/data/bc5cdr-c_v1/processed/val/neighbor_labels.npy"
    val_neighbor_labels = np.load(path)
    #reranker = reranker.to(device)
    best_val = 0
    best_model = None
    output = params['output_path']
    #n_candidates = params['n_candidates']
    model = model.to(device)
    criterion = torch.nn.BCELoss()
    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        model.train()
        logger.info("Training")
        #torch.cuda.empty_cache()
        gradient_accumulation_steps = 16
        loss_list = []
        correct = 0
        correct_5 = 0
        total_samples = 0
        
        for step, batch in enumerate(tqdm(train_dataloader, desc = "Processing minibatches")):
            #optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            mention_idxs, context_ids = batch    
            candidate_ids = []
            #print(mention_idxs.shape, context_ids.shape)
            # for idx in label_idxs:
            #     candidate_ids.append([])
      
            #     for neighbor in train_neighbors[idx]:
   
            #         candidate_ids[-1].append(train_entities[neighbor]['ids'])
            train_labels = []
            for idx in mention_idxs:
                candidate_ids.append([])
                train_labels.append(train_neighbor_labels[idx])
                
                for neighbor in train_neighbors[idx]:
                    candidate_ids[-1].append(train_entities[neighbor]['biobert_ids'])
                    
            
            #context_ids = context_ids.to(device)
            candidate_ids = torch.tensor(candidate_ids).to(device)
            #print("candidate_ids shape",candidate_ids.shape)
            scores = reranker(context_ids, candidate_ids, device) 
            total_samples += scores.shape[0]
            ## for binary cross entropy loss
            #train_labels = np.array(train_labels)
            train_labels = torch.FloatTensor(train_labels)
                
            # target = torch.zeros_like(scores)
            # target[:,0] = 1
            # target = target.float().to(device)
            #print(train_labels.shape)
            probs = F.softmax(scores, dim=1)
            loss =criterion(probs, train_labels.to(device))
            ordered_labels = []
            for scores_row, label_row in zip(scores, train_labels):
                ordered = np.array([x for _,x in sorted(zip(scores_row, label_row),reverse=True)])
                ordered_labels.append(ordered)
            #ordered_labels = 
            ordered_labels = torch.FloatTensor(ordered_labels)
            
            max_idxs = scores.argmax(dim=1)
            #correct = calculate_accuracy(max_idxs, correct)
            correct, correct_5 = calculate_top1_5(ordered_labels, correct, correct_5)
            
            loss = loss/gradient_accumulation_steps

            
            if params['data_parallel']:
                loss.mean().backward()
            else:
                loss.backward()
                
            if (step+1) % gradient_accumulation_steps == 0 or (step+1) == len(train_dataloader):
                loss_list.append(loss.item())
                logger.info("step: {}, accuracy1: {}, accuracy5: {}".format(step, correct/total_samples, correct_5/total_samples))

                optimizer.step()
                optimizer.zero_grad()
                
                
        logger.info("Training completed, with total samples: {}".format(total_samples))    
        logger.info("Training, epoch: {}, loss_list: {}, epoch_loss: {}, accuracy1: {}, accuracy5: {}".format(epoch_idx,loss_list, sum(loss_list)/len(loss_list), correct/total_samples, correct_5/total_samples))
        writer.add_scalar("Trainining loss/epoch 64 candidates", sum(loss_list)/len(loss_list), epoch_idx)
        
        writer.add_scalar("Training accuracy/epoch 64 candidates", correct/total_samples, epoch_idx)
        logger.info("Evaluating") 
        validation_loss, validation_accuracy, validation_accuracy5 = evaluate(reranker,val_dataloader, criterion, val_entities,val_neighbors, val_neighbor_labels,device, logger)

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
    
    path = "/share/project/biomed/hcd/Masked_EL/data/bc5cdr-c_v1/processed/test/biobert_dict.pkl"
    test_entities = load_pickle(path)
    path = "/share/project/biomed/hcd/Masked_EL/data/bc5cdr-c_v1/processed/test/id2idx.pkl"
    
    test_id2idx = load_pickle(path)
    
    path = "/share/project/biomed/hcd/Masked_EL/data/bc5cdr-c_v1/processed/test/neighbors.npy"
    test_neighbors = np.load(path)
    path = "/share/project/biomed/hcd/Masked_EL/data/bc5cdr-c_v1/processed/test/neighbor_labels.npy"
    test_neighbor_labels = np.load(path)
    test_tensor_data = process_mention_data(test_data, tokenizer, test_id2idx, params["debug"])
    test_dataloader = DataLoader(test_tensor_data, batch_size = val_test_batch_size, shuffle = False)
    test_loss, test_accuracy, test_accuracy5 = evaluate(reranker,test_dataloader, criterion, test_entities, test_neighbors, test_neighbor_labels,device, logger)
    logger.info("Evaluation loss: {}, Accuracy: {}, Accuracy5: {}".format(test_loss, test_accuracy, test_accuracy5))

    writer.flush()
    writer.close()
    
    
    
def eval_test(params):
    print("evaluating")
    reranker = BiEncoderRanker(params)
    logger = setup_logger('biencoder_logger', './logs/'+params['output_path']+'_biencoder64.log')
    tokenizer = reranker.tokenizer
    device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    path = "/share/project/biomed/hcd/Masked_EL/data/bc5cdr-c_v1/processed/test/biobert_dict.pkl"
    test_entities = load_pickle(path)
    path = "/share/project/biomed/hcd/Masked_EL/data/bc5cdr-c_v1/processed/test/id2idx.pkl"
    
    test_id2idx = load_pickle(path)
    
    path = "/share/project/biomed/hcd/Masked_EL/data/bc5cdr-c_v1/processed/test/neighbors.npy"
    test_neighbors = np.load(path)
    path = "/share/project/biomed/hcd/Masked_EL/data/bc5cdr-c_v1/processed/test/neighbor_labels.npy"
    test_neighbor_labels = np.load(path)
    model = reranker.model
    model = model.to(device)
    test_data = load_data_split("./data/bc5cdr-c_v1/processed/test.jsonl")
    val_test_batch_size = params["val_test_batch_size"]
    criterion = torch.nn.BCELoss()
    test_tensor_data = process_mention_data(test_data, tokenizer, test_id2idx,params["debug"])
    test_dataloader = DataLoader(test_tensor_data, batch_size = val_test_batch_size, shuffle = False)
    
    test_loss, test_accuracy, test_accuracy5 = evaluate(reranker,test_dataloader, criterion, test_entities, test_neighbors, test_neighbor_labels,device, logger)

    print(test_loss, test_accuracy, test_accuracy5)
    
if __name__ == "__main__":
    parameters = {"n_candidates":64, "val_test_batch_size":128, "gradient_accumulation_steps":16, "debug":False, "learning_rate":1e-6, "bert_model":"/share/project/biomed/hcd/arboEL/models/biobert-base-cased-v1.1", "out_dim":768, "epoch":100, "output_path":"test", "batch_size":8,  "n_gpu":1, "contrastive":False, "pairwise":False, "data_parallel":False, 'evaluation':True, "ckpt_path":"/share/project/biomed/hcd/Masked_EL/model_ckpts/test/best_model.pt"}
    if not parameters['evaluation']:
        main(parameters)
    else:
        print("evaluating")
        eval_test(parameters)
