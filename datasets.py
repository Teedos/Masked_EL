import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np

class UMLSKGDataset(Dataset):
    def __init__(self, triples_file):
        # Load the UMLS KG as a pandas dataframe
        self.kg_df = pd.read_csv(triples_file, sep="\t", header=None, names=["head", "relation", "tail"])

        # Convert the KG to the required format
        self.entities = sorted(set(self.kg_df["head"].unique()) | set(self.kg_df["tail"].unique()))
        self.entity_to_id = {entity: idx for idx, entity in enumerate(self.entities)}
        self.relations = sorted(self.kg_df["relation"].unique())
        self.relation_to_id = {relation: idx for idx, relation in enumerate(self.relations)}
        
        self.kg_df["head_id"] = self.kg_df["head"].apply(lambda x: self.entity_to_id[x])
        self.kg_df["tail_id"] = self.kg_df["tail"].apply(lambda x: self.entity_to_id[x])
        self.kg_df["relation_id"] = self.kg_df["relation"].apply(lambda x: self.relation_to_id[x])
        # Get the triples as numpy arrays
        self.triples = self.kg_df[["head_id", "relation_id", "tail_id"]].values.astype('int64')

        # Calculate the frequency of each entity in the head and tail positions
        self.entity_freq = np.zeros(len(self.entities))
        for head, _, tail in self.triples:
            self.entity_freq[head] += 1
            self.entity_freq[tail] += 1

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        # Generate negative sample
        head, relation, tail = self.triples[idx]
        corrupt_tail = tail
        while corrupt_tail == tail:
            # Randomly select an entity to corrupt the tail
            corrupt_tail = np.random.choice(len(self.entities), p=self.entity_freq / np.sum(self.entity_freq))
        negative_sample = torch.LongTensor([head, relation, corrupt_tail])

        return torch.LongTensor(self.triples[idx]), negative_sample
    

class PretrainDataset(Dataset):
    def __init__(self, name_ids, name_attn_mask, definition_ids, definition_attn_mask, index_tensor):
        #self.cuis = cui_list
        self.name_ids = name_ids  # Initialize the name_ids attribute
        self.definition_ids = definition_ids
        self.name_attn_mask = name_attn_mask
        self.definition_attn_mask = definition_attn_mask
        self.indexes = index_tensor

    def __len__(self):
        return len(self.name_ids)

    def __getitem__(self, index):
        #cui = self.cuis[index]
        names = self.name_ids[index]
        definition_ids = self.definition_ids[index]
        name_masks = self.name_attn_mask[index]
        definition_masks = self.definition_attn_mask[index]
        idxs = self.indexes[index]
        return  names,name_masks, definition_ids, definition_masks, idxs
    