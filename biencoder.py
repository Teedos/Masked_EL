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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.params['learning_rate'])

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
    
    def forward(self, context_ids, candidate_ids, target = None):
        
        try:
            context_emb  = self.encode_context(context_ids) # batch x emb size
            context_emb = context_emb.unsqueeze(2) # batch x emb size x 1
        except:
            print("context ids shape: ", context_ids.shape, context_ids)            
            print("candidate ids shape: ", candidate_ids.shape)
            
        b, n, s = candidate_ids.shape
        candidate_ids = candidate_ids.reshape(b*n,s)
        #print("candidate ids shape after",candidate_ids.shape)
        candidate_emb = self.encode_candidate(candidate_ids)    # batch x topk x emb size
        #print("candidate embedding shape before",candidate_emb.shape)
        candidate_emb = candidate_emb.reshape(b, n, 768)
        #print("candidate embeddings shape after",candidate_emb.shape)

        scores = torch.bmm(candidate_emb, context_emb)
        return context_emb.cpu(), candidate_emb.cpu(), scores
        
        
    