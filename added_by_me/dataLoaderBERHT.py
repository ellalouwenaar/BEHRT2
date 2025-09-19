import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

class BEHRTDataset(Dataset):
    def __init__(self, df, token2idx, age2idx, max_len=100):
        self.data = df
        self.token2idx = token2idx
        self.age2idx = age2idx
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        codes = row['code'][:self.max_len-1]  # -1 voor CLS
        ages = row['age'][:self.max_len-1]
        
        # Voeg CLS toe aan begin
        codes = np.concatenate([['CLS'], codes])
        ages = np.concatenate([[ages[0]], ages])
        
        # Converteer naar indices
        input_ids = [self.token2idx.get(code, self.token2idx['UNK']) for code in codes]
        age_ids = [self.age2idx.get(age, self.age2idx['UNK']) for age in ages]
        
        # Maak position en segment ids
        position_ids = list(range(len(input_ids)))
        segment_ids = self._create_segment_ids(codes)
        
        # Padding
        padding_length = self.max_len - len(input_ids)
        if padding_length > 0:
            input_ids += [self.token2idx['PAD']] * padding_length
            age_ids += [self.age2idx['PAD']] * padding_length
            position_ids += [0] * padding_length
            segment_ids += [0] * padding_length
        
        # Attention mask
        attention_mask = [1] * len(codes) + [0] * padding_length
        
        return {
            'input_ids': torch.tensor(input_ids[:self.max_len]),
            'age_ids': torch.tensor(age_ids[:self.max_len]),
            'position_ids': torch.tensor(position_ids[:self.max_len]),
            'segment_ids': torch.tensor(segment_ids[:self.max_len]),
            'attention_mask': torch.tensor(attention_mask[:self.max_len]),
            'patient_id': row['patid']
        }
    
    def _create_segment_ids(self, codes):
        """Wissel tussen 0 en 1 voor elke SEP"""
        segment_ids = []
        current_segment = 0
        for code in codes:
            segment_ids.append(current_segment)
            if code == 'SEP':
                current_segment = 1 - current_segment
        return segment_ids