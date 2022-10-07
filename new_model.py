import torch
import torch.nn as nn
import numpy as np
from SRFR_model import PointWiseFeedForward

##

class SRFR_with_BERT_Embedding(nn.Module):
    ## Embedding Extends item_features with fake/real review discriminations
    def __init__(self, item_number, item_embedding_size, fake_embedding_size, dropout_rate, maxlen, device):
        super(SRFR_Embedding,self).__init__()
        self.item_embed = nn.Embedding(item_number+1, item_embedding_size, padding_idx=0)
        self.fake_embed = nn.Embedding(3, fake_embedding_size, padding_idx = 0) #0 padding, 1 fake, 2 real
        self.pos_embed  = nn.Embedding(maxlen, item_embedding_size)
        self.total_hiden_size = item_embedding_size + fake_embedding_size
        self.dropout = nn.Dropout(p=dropout_rate)
        self.device = device
        
    def forward(self, input_ids, fake_ids=None):     
        
        #print(input_ids.shape)
        batch_size, seq_size = input_ids.shape[:2]
        
        input_embed = self.item_embed(input_ids) 
        pos_ids = torch.tile(torch.LongTensor(range(seq_size)), [batch_size, 1]).to(self.device)
        pos_embed = self.pos_embed(pos_ids)
        input_embed += pos_embed
        
        if fake_ids is None:
            fake_ids = torch.zeros(batch_size,seq_size).to(self.device).int()
        
        #print(fake_ids)
        fake_embed = self.fake_embed(fake_ids)
        input_embed = torch.cat([input_embed,fake_embed], dim=2)
        
        return input_embed
        
class SRFR_with_BERT(nn.Module):
    def __init__(self,
        item_number,
        max_len = 20,
        item_embedding_size = 50,
        fake_embedding_size = 10,
        dropout_rate = 0.5,
        num_blocks = 2,
        num_heads = 1,
        device = 'cpu'
        ):
        super(SRFR, self).__init__()
        
        self.device = device
        self.total_hidden_size = item_embedding_size + fake_embedding_size
        
        self.embedding_layer = SRFR_Embedding(item_number, item_embedding_size, fake_embedding_size, dropout_rate, max_len, device)
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        
        
        self.last_conv = torch.nn.Conv1d(self.total_hidden_size, item_embedding_size, kernel_size = 1)
        self.last_layernorm = torch.nn.LayerNorm(item_embedding_size, eps=1e-8)
        
        for _ in range(num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(self.total_hidden_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(self.total_hidden_size,num_heads,dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.total_hidden_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.total_hidden_size, self.total_hidden_size, dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def forward(self, user_ids, input_ids, fake_ids, positive_ids = None, positive_fake_ids = None, negative_ids = None, negative_fake_ids = None): # for training        
        
        input_embeds  = self.embedding_layer(input_ids, fake_ids)
        
        batch_size, seq_length = input_embeds.size()[:2]
        
        empty_ids = (input_ids == 0)
        input_embeds  *= ~empty_ids.unsqueeze(-1)
        
        
        non_empty_length = input_embeds.size()[1]
        attention_mask   = ~torch.tril(torch.ones((seq_length, seq_length), dtype=torch.bool)).to(self.device)
        
        hidden_state = input_embeds
        
        #print(hidden_state.size())
        
        for i in range(len(self.attention_layers)):
            hidden_state = torch.transpose(hidden_state, 0, 1)
            Q = self.attention_layernorms[i](hidden_state)
            mha_outputs, _ = self.attention_layers[i](Q, hidden_state, hidden_state, attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            hidden_state = Q + mha_outputs
            hidden_state = torch.transpose(hidden_state, 0, 1)
            
            hidden_state = self.forward_layernorms[i](hidden_state)
            hidden_state = self.forward_layers[i](hidden_state)
            
            hidden_state *=  ~empty_ids.unsqueeze(-1)
        
        hidden_state = self.last_conv(hidden_state.transpose(-1,-2))
        hidden_state = self.last_layernorm(hidden_state.transpose(-1,-2))
        
        #print(hidden_state.size())
        
        pos_logits = None
        if positive_ids is not None :
            pos_embs = self.embedding_layer.item_embed(positive_ids).to(self.device)
            pos_logits = (hidden_state * pos_embs).sum(dim=-1)
        
        neg_logits = None    
        if negative_ids is not None :    
            neg_embs = self.embedding_layer.item_embed(negative_ids).to(self.device)
            neg_logits = (hidden_state * neg_embs).sum(dim=-1)
        
        
        #print(hidden_state.size(), pos_embs.size(), pos_logits.size())
        
        
        return hidden_state, pos_logits, neg_logits #[pos_pred, neg_pred]

    def predict(self, user_ids, input_ids, fake_ids, label): # for inference
        hidden_state, pos_logits, neg_logits = self.forward(user_ids, input_ids, fake_ids) 
        
        final_features = hidden_state[:, -1, :]
        label_embeds = self.embedding_layer.item_embed(label).to(self.device) # (U, I, C)
        
        logits = label_embeds.matmul(final_features.unsqueeze(-1)).squeeze()
        
        return logits # preds # (U, I)    
