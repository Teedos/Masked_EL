import os
import torch
from transformers import BertModel, BertConfig, BertTokenizer
import torch.nn.functional as F

WEIGHT_DECAY = 0.001
class BiEncoderModule(torch.nn.Module):
    def __init__(self, params):
        
        super(BiEncoderModule, self).__init__()
        self.candidate_config = BertConfig.from_pretrained(params["bert_model"])
        self.candidate_config.max_position_embeddings = 25
        self.context_encoder = BertModel.from_pretrained(params["bert_model"]) # Could be a path containing config.json and pytorch_model.bin; or could be an id shorthand for a model that is loaded in the library
        self.candidate_encoder = BertModel(self.candidate_config)

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
    def __init__(self, params, path=None):
        super(BiEncoderRanker, self).__init__()
        self.params = params
        # self.device = torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu"
        # )
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(params['bert_model'],'vocab.txt'))
        self.n_gpu = params['n_gpu']
        self.model = BiEncoderModule(self.params)
        
        if params['evaluation']:
            print("building modellll")
            self.build_model()
        self.emb_size = 768
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.params['learning_rate']*self.params['gradient_accumulation_steps'])    
        if params['contrastive']:
            self.linear_layer = torch.nn.Linear(2*self.emb_size, 1).to(self.device)  #### learnable linear layer
            
        
        if params['data_parallel']:
            self.model = torch.nn.DataParallel(self.model)

        
    def build_model(self):     
        checkpoint = torch.load(self.params['ckpt_path'])
        self.model.load_state_dict(checkpoint['model'])
        
    def encode_context(self, context_ids):
        context_embedding, _ = self.model(context_ids=context_ids)
        #print(context_embedding)
        return context_embedding
    
    def encode_candidate(self, candidate_ids):
        _, candidate_embedding = self.model(candidate_ids=candidate_ids)
        return candidate_embedding
    
    def forward(self, context_ids, candidate_ids, device):
        
        context_emb = self.encode_context(context_ids)  # batch x token length
        if context_emb.ndim == 1:
            context_emb = context_emb.unsqueeze(0)
        context_emb = context_emb.unsqueeze(2)  # batch x emb size x 1

        b, n, s = candidate_ids.shape
        candidate_ids = candidate_ids.reshape(b * n, s) #batch x topk x token length
        candidate_emb = self.encode_candidate(candidate_ids)   # batch x topk x embed size
        candidate_emb = candidate_emb.reshape(b, n, self.emb_size)

        if self.params['pairwise']:
        ### dot product and cos similarity
            dot_product_scores = torch.bmm(candidate_emb,context_emb).squeeze(-1)
            norm_product = torch.norm(candidate_emb, dim=-1) * torch.norm(torch.transpose(context_emb, 1, 2), dim=-1)
            cosine_similarity_scores = dot_product_scores / norm_product
            scores = self.linear_layer(torch.cat((dot_product_scores.unsqueeze(-1), cosine_similarity_scores.unsqueeze(-1)), dim=-1)).squeeze(-1)
        else:
            scores = torch.bmm(candidate_emb, context_emb) # batch x topk x 1
            scores = scores.squeeze(dim=2)  ### batch x topk
        #print(scores.shape)
        return  scores
        
        
class PairwiseRankingLoss(torch.nn.Module):
    def __init__(self, margin=1):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin
    def forward(self, scores, labels):
        mask = labels.unsqueeze(1) != labels.unsqueeze(2)
        pos_scores = scores.unsqueeze(1).expand(-1, scores.size(0), -1)
        neg_scores = scores.unsqueeze(2).expand(-1, -1, scores.size(0))
        losses = torch.relu(neg_scores - pos_scores + self.margin) * mask.float()
        return losses.sum() / mask.sum()
    
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
