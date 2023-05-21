import torch
from tqdm import tqdm, trange
import logging
from biencoder_contrastive import BiEncoderModule, ContrastiveLoss
from process_pretrain import process_data
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import pickle
import os
import random

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

def load_data_split(path, debug= False):
    cuis = []
    names = []
    definitions = []
    with open(path,'r') as f:
        for line in f:
            line = line.strip()
            lst = line.split('\t')
            cui = lst[0]
            name = lst[1]
            definition = lst[2]
            cuis.append(cui)
            names.append(name.lower())
            definitions.append(definition.lower())
    if debug:
        return cuis[:1000], names[:1000], definitions[:1000]
    return cuis, names, definitions


def train_eval(params, train_dataloader, val_dataloader):
    model = BiEncoderModule(params)
    
    #self.n_gpu = params['n_gpu']
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    
    batch_size = params["batch_size"]
    model.to(device)
    train_epochs = params["epochs"]
    for epoch_idx in trange(int(train_epochs), desc="Training"):
        model.train()
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_dataloader, desc = "Processing batches")):
            #print(batch)
            name_ids,name_mask, definition_ids, definition_mask, indexes = batch
            
            context_embs, candidate_embs = model(definition_ids.to(device), definition_mask.to(device), name_ids.to(device), name_mask.to(device))
            b_size = context_embs.shape[0]
            negative_indices = [random.choice(list(range(b_size))) for _ in range(b_size)]
            negative_embeddings = candidate_embs[negative_indices]
            labels = torch.tensor([1] * b_size, dtype=torch.float)
            labels[torch.arange(b_size), negative_indices] = 0

            contrastive_loss = ContrastiveLoss()
            loss = contrastive_loss(context_embs, candidate_embs, labels) + \
                contrastive_loss(context_embs, negative_embeddings, 1 - labels)

            loss.backward()
            optimizer.step()

            print("loss", loss.item())

def main(params):
    tokenizer = BertTokenizer.from_pretrained(os.path.join(params['bert_model'],'vocab.txt'))
    debug = params["debug"]
    ### write
    # train_path = "/data/hcd/work_dir/Masked_EL/data/pretrain/sub_train_dictionary.txt"
    
    # train_cuis, train_names, train_definitions = load_data_split(train_path, debug)
    # #print(len(train_cuis), len(train_names), len(train_definitions))
    # val_path = "/data/hcd/work_dir/Masked_EL/data/pretrain/sub_val_dictionary.txt"
    # val_cuis,  val_names, val_definitions = load_data_split(val_path, debug)
    # #print(len(val_cuis), len(val_names), len(val_definitions))
    # train_samples, train_dataset = process_data(train_cuis, train_names, train_definitions, tokenizer= tokenizer)
    # with open('./data/pretrain/train/sub_train_samples.pickle','wb') as handler:
    #     pickle.dump(train_samples, handler)
    # torch.save(train_dataset,'./data/pretrain/train/sub_train_dataset.pt')
    # val_samples, val_dataset = process_data(val_cuis, val_names, val_definitions, tokenizer = tokenizer)
    # with open('./data/pretrain/val/sub_val_samples.pickle','wb') as handler:
    #     pickle.dump(val_samples, handler)
    # torch.save(val_dataset,'./data/pretrain/val/sub_val_dataset.pt')


    ####read 
    batch_size = params["batch_size"]
    train_dataset_path = "/data/hcd/work_dir/Masked_EL/data/pretrain/train/sub_train_dataset.pt"
    train_dataset = torch.load(train_dataset_path)
    #print(train_dataset[0].definition_ids)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    
    val_dataset_path = "/data/hcd/work_dir/Masked_EL/data/pretrain/val/sub_val_dataset.pt"
    val_dataset = torch.load(val_dataset_path)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle = True)
    

    ### Test data, ignore for now

    # test_path = "/data/hcd/work_dir/Masked_EL/data/pretrain/sub_test_dictionary.txt"
    # test_cuis, test_names, test_definitions = load_data_split(test_path, debug)
    # test_samples, test_dataset = process_data(test_cuis, test_names, test_definitions, tokenizer= tokenizer)
    # with open('./data/pretrain/test/sub_test_samples.pickle','wb') as handler:
    #     pickle.dump(test_samples, handler)
    # torch.save(test_dataset,'./data/pretrain/test/sub_test_dataset.pt')

    ##### read 
    test_dataset_path = "/data/hcd/work_dir/Masked_EL/data/pretrain/test/sub_test_dataset.pt"
    test_dataset = torch.load(test_dataset_path)
    print(len(train_dataset),len(val_dataset),len(test_dataset))

    train_eval(params, train_dataloader, val_dataloader)
    

if __name__ == "__main__":
    parameters = {"epochs":10, "learning_rate":1e-5,"debug":False, "bert_model":"/data/hcd/work_dir/pretrained_models/biobert-v1.1", "out_dim":768, "output_path":"pretrain_biencoder", "batch_size":512,  "n_gpu":1, 'evaluation':False}
    main(parameters)