import torch
import torch.nn as nn
import numpy as np
bce_criterion = torch.nn.BCEWithLogitsLoss()

class SRFR_Embedding(nn.Module):
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

class PointWiseFeedForward(nn.Module):
    def __init__(self, in_channel, out_channel, pwff_dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=1)
        self.dropout1 = nn.Dropout(p=pwff_dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size=1)
        self.dropout2 = nn.Dropout(p=pwff_dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class SRFR(nn.Module):
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

class SRFRN(nn.Module):
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
        super(SRFRN, self).__init__()
        
        self.device = device
        self.total_hidden_size = item_embedding_size + fake_embedding_size
        
        self.embedding_layer = SRFR_Embedding(item_number, item_embedding_size, fake_embedding_size, dropout_rate, max_len, device)
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(self.total_hidden_size, eps=1e-8)
        
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

        hidden_state = self.last_layernorm(hidden_state)
        
        pos_logits = None
        if positive_ids is not None :
            pos_embedding = torch.cat([self.embedding_layer.item_embed(positive_ids), self.embedding_layer.fake_embed(positive_fake_ids)], dim=2)
            pos_embedding.to(self.device)
            pos_logits = (hidden_state * pos_embedding).sum(dim=-1)
        
        neg_logits = None    
        if negative_ids is not None :
            neg_embedding = torch.cat([self.embedding_layer.item_embed(negative_ids), self.embedding_layer.fake_embed(negative_fake_ids)], dim=2)
            neg_embedding.to(self.device)  
            neg_logits = (hidden_state * neg_embedding).sum(dim=-1)
        
        
        #print(hidden_state.size(), pos_embs.size(), pos_logits.size())
        
        
        return hidden_state, pos_logits, neg_logits #[pos_pred, neg_pred]

    def predict(self, user_ids, input_ids, fake_ids, label): # for inference
        hidden_state, pos_logits, neg_logits = self.forward(user_ids, input_ids, fake_ids) 
        
        user_label = (torch.sign(torch.count_nonzero(fake_ids == 1, dim = 1) - torch.count_nonzero(fake_ids== 2, dim = 1))*0.5+1.5).int()
        user_label = torch.tile(user_label,(len(label),1)).squeeze()
        #extended_fake_ids = torch.mul(torch.sign(fake_ids),torch.roll(fake_ids,-1))
        #extended_fake_ids[:,-1] = user_label
        
        final_features = hidden_state[:, -1, :]
        
        item_embed   = self.embedding_layer.item_embed(label).to(self.device)
        review_embed = self.embedding_layer.fake_embed(user_label).to(self.device)
        #print(item_embed.size(),review_embed.size())
        
        label_embeds = torch.cat([item_embed,review_embed], dim=1) # (U, I, C)
        
        logits = label_embeds.matmul(final_features.unsqueeze(-1)).squeeze()
        
        return logits # preds # (U, I)
        

"""
class SRFRNudge_Embedding(nn.Module):
    ## Embedding Extends item_features with fake/real review discriminations
    def __init__(self, item_number, item_embedding_size, dropout_rate, maxlen, device):
        super(SRFRNudge_Embedding,self).__init__()
        self.item_embed = nn.Embedding(item_number+1, item_embedding_size, padding_idx=0)
        self.fakeness_embed = nn.Embedding(maxlen*2+1, item_embedding_size) #intensity of fakeness, from 0 most fakes, maxlen*2 least fake(real), maxlen as netural 
        self.pos_embed  = nn.Embedding(maxlen, item_embedding_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.device = device
        self.item_embedding_size = item_embedding_size
        self.maxlen = maxlen
        
    def forward(self, input_ids, fake_ids):
        #print(input_ids.shape)
        batch_size, seq_size = input_ids.shape[:2]
        
        input_embed = self.item_embed(input_ids) 
        pos_ids = torch.tile(torch.LongTensor(range(seq_size)), [batch_size, 1]).to(self.device)
        pos_embed = self.pos_embed(pos_ids)
        input_embed += pos_embed
        
        user_label = torch.count_nonzero(fake_ids == 1, dim = 1) - torch.count_nonzero(fake_ids== 2, dim = 1) + self.maxlen
        user_embed = self.fakeness_embed(user_label.view(-1,1))
        
        input_embed = input_embed + user_embed
        
        return input_embed
    
    def get_fakeness_embed(self):
        return self.fakeness_embed
        
class SRFRNudge_SASRec(nn.Module):
    ## Discriminate User with fake/real review counts
    def __init__(self,
        item_number,
        max_len = 20,
        item_embedding_size = 50,
        dropout_rate = 0.5,
        num_blocks = 2,
        num_heads = 1,
        device = 'cpu'
        ):
        super(SRFRNudge_SASRec, self).__init__()
        
        self.device = device
        self.maxlen = max_len
        
        self.embedding_layer = SRFRNudge_Embedding(item_number, item_embedding_size, dropout_rate, max_len, device)
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

    def forward(self, user_ids, input_ids, fake_ids, positive_ids = None, positive_fake_ids = None, negative_ids = None, negative_fake_ids = None): # for training        
        #print(input_ids[0],positive_ids[0],negative_ids[0])
        input_embeds  = self.embedding_layer(input_ids, fake_ids)
        
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
"""        
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
        
class SRFU(nn.Module):
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


class SRFU_B(SRFU):
    #Binary Classification
    
    def get_Labels(self, fake_ids):
        
        user_label_ids = (torch.sign(torch.count_nonzero(fake_ids == 1, dim = 1) - torch.count_nonzero(fake_ids== 2, dim = 1))*0.5+1.5)
        user_label_ids = torch.round(user_label_ids).int()
        
        return user_label_ids
        
class SRFU_F(SRFU):
    #Frequency Classification
    
    def get_Labels(self, fake_ids):
        
        user_label_ids = torch.count_nonzero(fake_ids == 1, dim = 1)
        
        return user_label_ids

class SRFU_R(SRFU):
    #Ratio Classification
    
    def get_Labels(self, fake_ids):
        
        user_label_ids = torch.count_nonzero(fake_ids == 1, dim = 1) / (torch.count_nonzero(fake_ids == 1, dim = 1) + torch.count_nonzero(fake_ids== 2, dim = 1)) * 10
        user_label_ids = torch.floor(user_label_ids).int()
        
        return user_label_ids

class SASRec(torch.nn.Module):
    def __init__(self,
        item_number,
        maxlen = 20,
        hidden_units = 50,
        dropout_rate = 0.5,
        num_blocks = 2,
        num_heads = 1,
        device = 'cpu'
        ):
        super(SASRec, self).__init__()

        #self.user_num = user_num
        self.item_num = item_number
        self.dev = device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(maxlen, hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(hidden_units,
                                                            num_heads,
                                                            dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(hidden_units, hidden_units, dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):
        
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = (log_seqs == 0)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, input_ids, fake_ids, positive_ids = None, positive_fake_ids = None, negative_ids = None, negative_fake_ids = None): # for training        
        #print(input_ids,positive_ids,negative_ids)
        #raise('stop')
        
        log_feats = self.log2feats(input_ids) # user_ids hasn't been used yet

        pos_embs = self.item_emb(positive_ids)
        neg_embs = self.item_emb(negative_ids)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return log_feats, pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, fake_ids, item_indices): # for inference
        #print(log_seqs,fake_ids,item_indices)
        #raise('stop')
        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(item_indices) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze()
        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)  
