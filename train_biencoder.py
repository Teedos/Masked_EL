import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import numpy as np
from biencoder import BiEncoderRanker, PairwiseRankingLoss
from process_data import load_data_split, load_entities, process_entity_dictionary, process_mention_data, process_candidate_data, embed_dictionary
import pickle
import argparse
import torch.nn.functional as F
import logging
from transformers import BertModel, BertTokenizer
from torch.utils.tensorboard import SummaryWriter


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

def evaluate(reranker, val_dataloader, criterion, entity_dictionary, neighbor_list, device, logger, pairwise = False):
    reranker.model.eval()
    total_samples = 0
    correct = 0
    loss_list = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(val_dataloader,desc="validation mini batches")):
                torch.cuda.empty_cache()
            
                batch = tuple(t.to(device) for t in batch)
                context_ids, label_idxs = batch    
                candidate_ids = []
                for idx in label_idxs:
                    candidate_ids.append([])
        
                    for neighbor in neighbor_list[idx]:
    
                        candidate_ids[-1].append(entity_dictionary[neighbor]['ids'])
                
                candidate_ids = torch.tensor(candidate_ids).to(device)
                #print(candidate_ids.shape)
                scores = reranker(context_ids, candidate_ids, device) 
                total_samples += scores.shape[0]
                if pairwise:
                    target = [1] + [-1] * (len(label_idxs)-1)
                
                    target = torch.tensor(target).float().to(device)
                    
                else:
                    # target = [1] + [0] * (len(label_idxs)-1)
                
                    # target = torch.tensor(target).float().to(device)
                    target = torch.zeros_like(scores)
                    target[:,0] = 1
                    target = target.float().to(device)
                    max_idxs = scores.argmax(dim=1)
                    correct = calculate_accuracy(max_idxs, correct)
                    probs = F.softmax(scores, dim=1)

                    #p_correct = probs[:, 0]
                    
                    loss = criterion(probs, target)
                   
                    loss_list.append(loss.item())

    logger.info("Evaluation completed, with total samples: {}".format(total_samples))           
    return sum(loss_list)/len(loss_list), correct/total_samples

def main(params):
    writer = SummaryWriter()
    logger = setup_logger('biencoder_logger', './logs/biencoder64.log')
    reranker = BiEncoderRanker(params)
    model = reranker.model
    pretrained_biosyn = BertModel.from_pretrained('/share/project/biomed/hcd/BioSyn/pretrained/biosyn-sapbert-bc5cdr-chemical') 
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
    processed_val_mentions, val_tensor_data = process_mention_data(val_data, tokenizer, id_to_idx, params["debug"])
    batch_size = params['batch_size']
    val_test_batch_size = params["val_test_batch_size"]
    train_dataloader = DataLoader(train_tensor_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_tensor_data, batch_size = val_test_batch_size, shuffle = True)

    num_train_epochs = params['epoch']

    n_candidates = params['n_candidates']
    ent_embs, neighbor_list = embed_dictionary(pretrained_biosyn, device, entity_dictionary, n_candidates, params["debug"])

    entity_dictionary_pkl_path = './processed/entity_dictionary.pkl'
    
    with open(entity_dictionary_pkl_path, 'wb') as write_handle:
                    pickle.dump(entity_dictionary, write_handle,
                                protocol=pickle.HIGHEST_PROTOCOL)

    entity_embedding_pkl_path = './processed/entity_embedding.pkl'

    with open(entity_embedding_pkl_path, 'wb') as write_handle:
                    pickle.dump(ent_embs, write_handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
    
    np.save('./processed/neighbor_list_64.npy', neighbor_list)
    
    best_val = 0
    best_model = None
    pairwise = params['pairwise']
    output = params['output_path']
    #n_candidates = params['n_candidates']
    if pairwise:
        criterion = PairwiseRankingLoss()
    else:
        #criterion = F.binary_cross_entropy()
        criterion = torch.nn.BCELoss()
    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        model.train()
        logger.info("Training")
        torch.cuda.empty_cache()
        gradient_accumulation_steps = 16
        loss_list = []
        correct = 0
        total_samples = 0
        avg_train_accuracy = []
        preds = []
        for step, batch in enumerate(tqdm(train_dataloader, desc = "Processing minibatches")):
            #optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            context_ids, label_idxs = batch    
            candidate_ids = []
            for idx in label_idxs:
                candidate_ids.append([])
      
                for neighbor in neighbor_list[idx]:
   
                    candidate_ids[-1].append(entity_dictionary[neighbor]['ids'])
            
            candidate_ids = torch.tensor(candidate_ids).to(device)
            scores = reranker(context_ids, candidate_ids, device) 
            total_samples += scores.shape[0]
            ## for binary cross entropy loss
            if pairwise:
                target = [1] + [-1] * (len(label_idxs)-1)
            
                target = torch.tensor(target).float().to(device)
                loss = criterion(scores, target)
                preds = (scores > 0).int()
                train_accuracy = (preds == target).float().mean()
                avg_train_accuracy.append(train_accuracy)    
            else:
                
                target = torch.zeros_like(scores)
                target[:,0] = 1
                target = target.float().to(device)
                probs = F.softmax(scores, dim=1)
                loss =criterion(probs, target)
                max_idxs = scores.argmax(dim=1)
                correct = calculate_accuracy(max_idxs, correct)
           
            loss = loss/gradient_accumulation_steps

            
            if params['data_parallel']:
                loss.mean().backward()
            else:
                loss.backward()
                
            if (step+1) % gradient_accumulation_steps == 0 or (step+1) == len(train_dataloader):
                loss_list.append(loss.item())
                logger.info("step: {}, accuracy: {}".format(step, correct/total_samples))

                optimizer.step()
                optimizer.zero_grad()
                
                
        logger.info("Training completed, with total samples: {}".format(total_samples))    
        logger.info("Training, epoch: {}, loss_list: {}, epoch_loss: {}, accuracy: {}".format(epoch_idx,loss_list, sum(loss_list)/len(loss_list), correct/total_samples))
        writer.add_scalar("Trainining loss/epoch 64 candidates", sum(loss_list)/len(loss_list), epoch_idx)
        
        writer.add_scalar("Training accuracy/epoch 64 candidates", correct/total_samples, epoch_idx)
        logger.info("Evaluating") 
        validation_loss, validation_accuracy = evaluate(reranker,val_dataloader, criterion, entity_dictionary,neighbor_list, device, logger)

        logger.info("Validation, epoch: {}, loss: {}, accuracy: {}".format(epoch_idx, validation_loss, validation_accuracy))
        
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
    processed_test_mentions, test_tensor_data = process_mention_data(test_data, tokenizer, id_to_idx, params["debug"])
    test_dataloader = DataLoader(test_tensor_data, batch_size = val_test_batch_size, shuffle = False)
    test_loss, test_accuracy = evaluate(reranker,test_dataloader, criterion, entity_dictionary, neighbor_list, device, logger)
    logger.info("Evaluation loss: {}, Accuracy: {}".format(test_loss, test_accuracy))
    # writer.add_scalar("Test loss", test_loss)
    # writer.add_scalar("Test accuracy", test_accuracy)
    
    writer.flush()
    writer.close()
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # Input data
    parser.add_argument(
        "--batch_size",
        default="8",
        type=int,
        required=False,
    )

    parser.add_argument(
        "--output_path",
        default="64_candidates",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--epoch",
        default=100,
        type=int,
        required=False,
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
        default="/share/project/biomed/hcd/arboEL/models/biobert-base-cased-v1.1",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--learning_rate",
        default=1e-6,
        type=int,
        required=False,
    )

    parser.add_argument(
        "--debug",
        type=bool,
        default = False,
        required=False,
    )
    
    parser.add_argument(
        "--data_parallel",
        type=bool,
        default=False,
        required=False,
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        required=False,
    )
    
    parser.add_argument(
        "--val_test_batch_size",
        type=int,
        default=128,
        required=False,
    )
    
    parser.add_argument(
        "--pairwise",
        type=bool,
        default=False,
        required=False,
    )
    parser.add_argument(
        "--contrastive",
        type=bool,
        default=True,
        required=False,
    )
    
    parser.add_argument(
        "--n_candidates",
        type=int,
        default=64,
        required=False,
    )
    args = parser.parse_args()

    parameters = args.__dict__
    main(parameters)