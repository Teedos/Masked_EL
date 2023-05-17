import torch
import torch.nn as nn
import torch.nn.functional as F

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, margin):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        #print(num_entities, embedding_dim)
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.initialize_embeddings()

    def initialize_embeddings(self):
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

    def forward(self, heads, relations, tails):
        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)
        projected_tail_embeddings = head_embeddings + relation_embeddings
        distances = F.pairwise_distance(projected_tail_embeddings, tail_embeddings)
        return distances

    def compute_loss(self, positive_samples, negative_samples):
        pos_heads, pos_relations, pos_tails = positive_samples[:, 0], positive_samples[:, 1], positive_samples[:, 2]
        neg_heads, neg_relations, neg_tails = negative_samples[:, 0], negative_samples[:, 1], negative_samples[:, 2]
        pos_distances = self.forward(pos_heads, pos_relations, pos_tails)
        neg_distances = self.forward(neg_heads, neg_relations, neg_tails)
        loss = torch.max(torch.zeros_like(neg_distances), self.margin + pos_distances - neg_distances)
        return loss.mean()