import os
import torch
from transformers import BertTokenizer
from transformers import BertModel
import torch.nn.functional as F

class BiEncoderModule(torch.nn.Module):
    def __init__(self, params):
        super(BiEncoderModule, self).__init__()
        self.context_encoder = BertModel.from_pretrained(params["bert_model"]) # Could be a path containing config.json and pytorch_model.bin; or could be an id shorthand for a model that is loaded in the library
        self.candidate_encoder = BertModel.from_pretrained(params["bert_model"])

    def forward(
        self,
        context_ids = None,
        candidate_ids = None
    ):
        context_embedding = None
        if context_ids is not None:
            context_embedding = self.context_encoder(context_ids)
            context_embedding = context_embedding[0].mean(1).squeeze(0)
        candidate_embedding = None
        if candidate_ids is not None:
            candidate_embedding = self.candidate_encoder(candidate_ids)
            candidate_embedding = candidate_embedding[0].mean(1).squeeze(0)
        return context_embedding, candidate_embedding
    
class BiEncoderRanker(torch.nn.Module):
    def __init__(self, params):
        super(BiEncoderRanker, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(params['bert_model'],'vocab.txt'))
        self.n_gpu = params['n_gpu']
        self.model = BiEncoderModule(self.params)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.params['learning_rate']*self.params['gradient_accumulation_steps'])
        self.emb_size = 768
    
        self.loss_fn = ContrastiveLoss(0.1)
        self.model = self.model.to(self.device)
        if params['data_parallel']:
            self.model = torch.nn.DataParallel(self.model)
            
    def encode_context(self, context_ids):
        context_embedding, _ = self.model(context_ids=context_ids)
        #print(context_embedding)
        return context_embedding
    
    def encode_candidate(self, candidate_ids):
        _, candidate_embedding = self.model(candidate_ids=candidate_ids)
        return candidate_embedding
    
    def forward(self, context_ids, candidate_ids):
        
        
        context_emb = self.encode_context(context_ids)  # batch x emb size
        context_emb = context_emb.unsqueeze(2)  # batch x emb size x 1

        b, n, s = candidate_ids.shape
        candidate_ids = candidate_ids.reshape(b * n, s)
        candidate_emb = self.encode_candidate(candidate_ids)  # batch x topk x emb size
        candidate_emb = candidate_emb.reshape(b, n, self.emb_size)

        context_emb = context_emb.permute(0, 2, 1)
        output1 = context_emb.expand(-1, n, -1)
        output2 = candidate_emb
        label = [0] + [1] * (n-1)
        label = torch.tensor(label).float().to(self.device)
        scores = -1 * self.loss_fn(
            output1, output2, label=label)
        
        return  scores
        
    
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        # loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
        #                               (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        loss_contrastive = (1-label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return loss_contrastive
