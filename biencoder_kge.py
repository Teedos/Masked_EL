import os
import torch
from transformers import BertModel, BertConfig, BertTokenizer
import torch.nn.functional as F
from transe import TransE
import numpy as np

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
    
class KG_Encoder(torch.nn.Module): 
    def __init__(self):
        super(KG_Encoder, self).__init__()
        #self.params = params
        self.n_entities = 81495
        self.n_rels = 193
        self.emb_size = 256
        self.margin = 1
        self.output_size = 128
        self.model = TransE(self.n_entities, self.n_rels, self.emb_size, self.margin)
        self.build_model()
        self.entity_embeddings = self.model.entity_embeddings
        self.embedding_layer = torch.nn.Linear(self.emb_size, self.output_size)
    def build_model(self):
        print("building kg encoder")
        model_state_dict = torch.load('/share/project/biomed/hcd/Masked_EL/model_ckpts/transE/best_model.pt') 
        self.model.load_state_dict(model_state_dict['model'])
    
    def forward(self, embeddings):
        #embeddings = embeddings.long()
        embeddings.detach()
        embs = self.embedding_layer(embeddings)
        embs = torch.relu(embs)
        return embs
    
class Reranker(torch.nn.Module):
    def __init__(self, params, path=None):
        super(Reranker, self).__init__()
        self.params = params
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(params['bert_model'],'vocab.txt'))
        self.n_gpu = params['n_gpu']
        self.model = BiEncoderModule(self.params)
        self.kg_encoder = KG_Encoder()
        
        if params['evaluation']:
            print("building modellll")
            self.build_model()
        self.bert_size = 768
        self.kg_size = 128
        self.emb_size = self.bert_size + self.kg_size
        
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
        return context_embedding
    
    def encode_candidate(self, candidate_ids):
        _, candidate_embedding = self.model(candidate_ids=candidate_ids)
        return candidate_embedding
    
    def encode_kg_entity(self, kg_embeddings):
        updated_kg_embeddings = self.kg_encoder(kg_embeddings)
        return updated_kg_embeddings
    
    def forward(self, context_ids, candidate_ids, batch_mesh_ids, mesh2cui, cui2idx, device):
        batch_kg_embeddings = []
        covered = 0
        not_covered = 0
        for row in batch_mesh_ids:
            #batch_kg_embeddings.append()
            embs = []
            for mesh in row:
                if mesh in mesh2cui:
                    cui = mesh2cui[mesh]
                    if cui in cui2idx:
                        covered +=1
                        kg_idx = cui2idx[cui]
                        kg_emb = self.kg_encoder.entity_embeddings(torch.tensor(kg_idx).to(device))   
                        #batch_kg_embeddings[-1].append(kg_emb)
                        embs.append(kg_emb)
                    else:
                        not_covered +=1
                        entity_embeddings = self.kg_encoder.entity_embeddings.weight.data
                        embs.append(torch.mean(entity_embeddings, dim = 0))
                else:
                    not_covered +=1
                    entity_embeddings = self.kg_encoder.entity_embeddings.weight.data
                    embs.append(torch.mean(entity_embeddings, dim = 0))
            batch_kg_embeddings.append(torch.stack(embs))
        batch_kg_embeddings = torch.stack(batch_kg_embeddings)
        kg_emb = self.encode_kg_entity(batch_kg_embeddings)
        kg_emb.to(device)
        b, n, s = candidate_ids.shape          
        context_emb = self.encode_context(context_ids)  
        if context_emb.ndim == 1:
            context_emb = context_emb.unsqueeze(0)
        context_emb = context_emb.unsqueeze(1).repeat(1, n, 1)    
        concatenated_context = torch.cat([context_emb, kg_emb], dim=2)
        
        candidate_ids = candidate_ids.reshape(b * n, s) 
        candidate_emb = self.encode_candidate(candidate_ids)   
        candidate_emb = candidate_emb.reshape(b, n, self.bert_size)
        concatenated_candidate = torch.cat((candidate_emb, kg_emb), dim=-1)
        if self.params['pairwise']:
            dot_product_scores = torch.bmm(candidate_emb,context_emb).squeeze(-1)
            norm_product = torch.norm(candidate_emb, dim=-1) * torch.norm(torch.transpose(context_emb, 1, 2), dim=-1)
            cosine_similarity_scores = dot_product_scores / norm_product
            scores = self.linear_layer(torch.cat((dot_product_scores.unsqueeze(-1), cosine_similarity_scores.unsqueeze(-1)), dim=-1)).squeeze(-1)
        else:
            scores = torch.bmm(concatenated_candidate, concatenated_context.transpose(1,2)) 
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
