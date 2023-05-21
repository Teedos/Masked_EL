import os
import torch
from transformers import BertModel, BertConfig, BertTokenizer
import torch.nn.functional as F
import torch.nn as nn
#from sklearn.metrics.pairwise import pairwise_distances_argmin_neg
WEIGHT_DECAY = 0.001
class BiEncoderModule(torch.nn.Module):
    def __init__(self, params):
        
        super(BiEncoderModule, self).__init__()
        self.candidate_config = BertConfig.from_pretrained(params["bert_model"])
        #self.candidate_config.max_position_embeddings = 25
        self.context_encoder = BertModel.from_pretrained(params["bert_model"]) # Could be a path containing config.json and pytorch_model.bin; or could be an id shorthand for a model that is loaded in the library
        self.candidate_encoder = BertModel.from_pretrained(params["bert_model"])

    def forward(
        self,
        context_ids = None,
        context_mask = None,
        candidate_ids = None,
        candidate_mask = None
    ):
        context_embedding = None
        if context_ids is not None:
            context_embedding = self.context_encoder(context_ids, context_mask)
            context_embedding = context_embedding[0].mean(1).squeeze(0)
        candidate_embedding = None
        if candidate_ids is not None:
            candidate_embedding = self.candidate_encoder(candidate_ids, candidate_mask)
            candidate_embedding = candidate_embedding[0].mean(1).squeeze(0)
        return context_embedding, candidate_embedding
    
    
# class BiEncoder(torch.nn.Module):
#     def __init__(self, params, path=None):
#         super(BiEncoder, self).__init__()
#         self.params = params
#         # self.device = torch.device(
#         #     "cuda" if torch.cuda.is_available() else "cpu"
#         # )
#         #self.tokenizer = BertTokenizer.from_pretrained(os.path.join(params['bert_model'],'vocab.txt'))
#         self.n_gpu = params['n_gpu']
#         self.model = BiEncoderModule(self.params)
        
#         if params['evaluation']:
#             print("building modellll")
#             self.build_model()
#         self.emb_size = 768
#         #   self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.params['learning_rate']*self.params['gradient_accumulation_steps'])    

#     def build_model(self):     
#         checkpoint = torch.load(self.params['ckpt_path'])
#         self.model.load_state_dict(checkpoint['model'])
        
#     def encode_context(self, context_ids, context_mask):
#         context_embedding, _ = self.model(context_ids=context_ids, context_mask= context_mask)
#         #print(context_embedding)
#         return context_embedding
    
#     def encode_candidate(self, candidate_ids, candidate_mask):
#         _, candidate_embedding = self.model(candidate_ids=candidate_ids, candidate_mask = candidate_mask)
#         return candidate_embedding
    
#     def forward(self, context_ids, context_mask, candidate_ids, candidate_mask,device):
        
#         context_embeddings = self.encode_context(context_ids, context_mask)  # batch x token length
#         # if context_emb.ndim == 1:
#         #     context_emb = context_emb.unsqueeze(0)
#         # context_emb = context_emb.unsqueeze(2)  # batch x emb size x 1
#         #context_embeddings 
#         #b, n, s = candidate_ids.shape
#         #candidate_ids = candidate_ids.reshape(b * n, s) #batch x topk x token length
#         candidate_embeddings = self.encode_candidate(candidate_ids, candidate_mask)   # batch x topk x embed size


        
#         return  loss
        
    

#     def forward(self, context_embeddings, candidate_embeddings, labels):
        

#         return loss_contrastive
# # Define contrastive loss function with hard negative mining
# class ContrastiveLoss(nn.Module):
#     def __init__(self, margin, hard_negative_ratio):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin
#         self.hard_negative_ratio = hard_negative_ratio
    
#     def forward(self, context_embeddings, candidate_embeddings, labels):
#         batch_size = context_embeddings.size(0)
        
#         # Compute cosine similarity between embeddings
#         similarity = nn.functional.cosine_similarity(context_embeddings, candidate_embeddings)
        
#         # Get hard negative samples using pairwise distances
#         num_hard_negatives = int(self.hard_negative_ratio * batch_size)
#         hard_negatives = pairwise_distances_argmin_neg(context_embeddings, candidate_embeddings, axis=1, n_jobs=-1)
        
#         # Calculate contrastive loss with hard negatives
#         positive_similarity = similarity.diag().unsqueeze(1)
#         hard_negative_similarity = similarity[torch.arange(batch_size), hard_negatives]
        
#         loss = torch.mean(torch.clamp(self.margin - positive_similarity + hard_negative_similarity, min=0))
#         return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, context_embeddings, candidate_embeddings, labels):
        distances = torch.nn.functional.pairwise_distance(context_embeddings, candidate_embeddings)

        # Calculate contrastive loss
        loss_contrastive = torch.mean((1 - labels) * torch.pow(distances, 2) +
                                      labels * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2))

        return loss_contrastive