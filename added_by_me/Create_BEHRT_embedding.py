import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

class BEHRTEmbeddings(nn.Module):
    def __init__(self, config, vocab_size, age_vocab_size):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.segment_embeddings = nn.Embedding(2, config.hidden_size)  # A/B segments
        self.age_embeddings = nn.Embedding(age_vocab_size, config.hidden_size)
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, input_ids, position_ids, segment_ids, age_ids):
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        segment_embeds = self.segment_embeddings(segment_ids)
        age_embeds = self.age_embeddings(age_ids)
        
        embeddings = word_embeds + position_embeds + segment_embeds + age_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BEHRT(nn.Module):
    def __init__(self, vocab_size, age_vocab_size, config=None):
        super().__init__()
        
        if config is None:
            config = BertConfig(
                vocab_size=vocab_size,
                hidden_size=288,
                num_hidden_layers=6,
                num_attention_heads=12,
                intermediate_size=512,
                max_position_embeddings=512
            )
        
        self.config = config
        self.embeddings = BEHRTEmbeddings(config, vocab_size, age_vocab_size)
        self.encoder = BertModel(config).encoder
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(self, input_ids, position_ids, segment_ids, age_ids, attention_mask):
        embeddings = self.embeddings(input_ids, position_ids, segment_ids, age_ids)
        
        # Pas attention mask aan voor BERT
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        encoder_outputs = self.encoder(embeddings, extended_attention_mask)
        sequence_output = encoder_outputs.last_hidden_state
        
        # Pool het CLS token (eerste positie)
        pooled_output = self.pooler(sequence_output[:, 0])
        
        return sequence_output, pooled_output