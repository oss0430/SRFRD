import torch
import torch.nn as nn
import numpy as np
from SRFR_model import PointWiseFeedForward

##

class SRFU_Embedding(nn.Module):
    ## Embedding Extends item_features with fake/real review discriminations
    def __init__(self, item_number, item_embedding_size, number_of_labels, dropout_rate, maxlen, device):
        super(SRFU_Embedding,self).__init__()
        self.item_embed = nn.Embedding(item_number+1, item_embedding_size, padding_idx=0)
        self.user_label_embed = nn.Embedding(number_of_labels, item_embedding_size) #intensity of fakeness, from 0 most fakes, maxlen*2 least fake(real), maxlen as netural 
        self.pos_embed  = nn.Embedding(maxlen, item_embedding_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.device = device
        self.item_embedding_size = item_embedding_size
        self.number_of_labels = number_of_labels
        self.maxlen = maxlen
        
    def forward(self, input_ids, label_ids):
        #print(input_ids.shape)
        batch_size, seq_size = input_ids.shape[:2]
        
        input_embed = self.item_embed(input_ids) 
        pos_ids = torch.tile(torch.LongTensor(range(seq_size)), [batch_size, 1]).to(self.device)
        pos_embed = self.pos_embed(pos_ids)
        input_embed += pos_embed
        
        user_embed = self.user_label_embed(label_ids.view(batch_size,-1))
        
        input_embed = input_embed + user_embed
        
        return input_embed
    
    def get_user_label_embed(self):
        return self.user_label_embed
        
class SRFU_with_BERT(nn.Module):
    ## Discriminate User with fake/real review counts
    def __init__(self,
        item_number,
        max_len = 20,
        item_embedding_size = 50,
        number_of_labels = 2,
        dropout_rate = 0.5,
        num_blocks = 2,
        num_heads = 1,
        device = 'cpu'
        ):
        super(SRFU, self).__init__()
        
        self.device = device
        self.maxlen = max_len
        
        self.embedding_layer = SRFU_Embedding(item_number, item_embedding_size, number_of_labels, dropout_rate, max_len, device)
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(item_embedding_size, eps=1e-8)
        
        for _ in range(num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(item_embedding_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(item_embedding_size,num_heads,dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(item_embedding_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(item_embedding_size, item_embedding_size, dropout_rate)
            self.forward_layers.append(new_fwd_layer)
    
    def get_Labels(self, fake_ids):
        
        user_label_ids = None
        
        raise('not implemented result')
        
        return user_label_ids
    
    def forward(self, user_ids, input_ids, fake_ids, positive_ids = None, positive_fake_ids = None, negative_ids = None, negative_fake_ids = None): # for training        
        #print(input_ids[0],positive_ids[0],negative_ids[0])
        input_embeds  = self.embedding_layer(input_ids, self.get_Labels(fake_ids))
        
        batch_size, seq_length = input_embeds.size()[:2]
        
        #print(input_embeds.size())
        
        empty_ids = (input_ids == 0)
        
        input_embeds  *= ~empty_ids.unsqueeze(-1)
        
        
        non_empty_length = input_embeds.size()[1]
        attention_mask   = ~torch.tril(torch.ones((seq_length, seq_length), dtype=torch.bool)).to(self.device)
        #since first element in sequence is user (length =  1 dont need chagning in mask) 
        
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
        
        hidden_state = self.last_layernorm(hidden_state)
        
        #print(hidden_state.size())
        
        pos_logits = None
        if positive_ids is not None :
            #print(positive_ids.size())
            pos_embs = self.embedding_layer.item_embed(positive_ids).to(self.device)
            pos_logits = (hidden_state * pos_embs).sum(dim=-1)
        
        neg_logits = None    
        if negative_ids is not None :    
            #print(negative_ids.size())
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