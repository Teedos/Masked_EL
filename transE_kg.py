import torch
from transe import TransE
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import logging 

from datasets import UMLSKGDataset


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


dataset = UMLSKGDataset("/data/hcd/work_dir/Masked_EL/data/UMLS/kg.txt")


def train_and_evaluate(model, train_dataloader, valid_dataloader, logger, num_epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True, eps=1e-5)
    device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    model.to(device)
    #model = torch.nn.DataParallel(model)

    train_losses = []
    val_losses = []
    best_model = None
    best_val = 0
    #print("I am here")
    for epoch in trange(int(num_epochs), desc="Epoch"):
        train_loss = 0.0
        model.train()
        for idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            positive_samples, negative_samples = batch
            optimizer.zero_grad()
            loss = model(positive_samples.to(device), negative_samples.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_dataloader)
        valid_loss = evaluate(model, valid_dataloader, device)
        #print("printing log")
        logger.info("Epoch {}, Train Loss: {}, Valid Loss: {}".format(epoch,train_loss,valid_loss))
        # Update learning rate scheduler
        scheduler.step(valid_loss)
        #best_val = valid_loss
        if best_val == 0:
            best_val = valid_loss
        if valid_loss < best_val:

            best_model = {'model': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'epoch': epoch}
            torch.save(best_model, './model_ckpts/transE/best_model.pt')
        if (epoch+1)%10 == 0:
            checkpoint = {'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch':epoch}
            torch.save(checkpoint, './model_ckpts/transE/model_ckpt_'+str(epoch)+'.pt')
    return train_losses, val_losses

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader,desc = "Validation")):
            positive_samples, negative_samples = batch
            loss = model(positive_samples.to(device), negative_samples.to(device))
            total_loss += loss.item()
    total_loss /= len(dataloader)
    return total_loss


n_entities = len(dataset.entities)
n_rels = len(dataset.relations)
n_embs = 256
margin = 1
model = TransE(n_entities, n_rels, n_embs, margin)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
logger = setup_logger('TransE_logger', './logs/transE.log')
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)
train_loss, val_loss = train_and_evaluate(model, train_dataloader, test_dataloader, logger, num_epochs=100, learning_rate=1e-4)

