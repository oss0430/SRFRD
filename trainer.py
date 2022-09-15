import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch import cuda
from SRFR_model import SASRec, SRFR, SRFRN, SRFU_B, SRFU_F, SRFU_R
from utils import *

import wandb
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

def simulate(model, optimizer, criterion, sampler, config, wandb, model_index=0):
    T = 0.0
    t0 = time.time()
    
    metricsbyepoch = dict()
    
    for epoch in range(config.num_epochs):
        if config.inference_only: break
        
        epoch_loss = 0
        model.train()
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, rsq, pos, prs, neg, nrs = sampler.next_batch() # tuples to ndarray
            u, seq, rsq, pos, prs, neg, nrs = np.array(u), np.array(seq), np.array(rsq), np.array(pos), np.array(prs), np.array(neg) , np.array(nrs)
            u, seq, rsq, pos, prs, neg, nrs = torch.LongTensor(u).to(device), torch.LongTensor(seq).to(device), torch.LongTensor(rsq).to(device), torch.LongTensor(pos).to(device), torch.LongTensor(prs).to(device), torch.LongTensor(neg).to(device), torch.LongTensor(nrs).to(device),
            hidden_state, pos_logits, neg_logits = model(user_ids = u , input_ids = seq, fake_ids = rsq, positive_ids = pos, positive_fake_ids=prs, negative_ids = neg, negative_fake_ids=nrs)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=device), torch.zeros(neg_logits.shape, device=device)
            #print("\neye ball check raw_logits:")
            #print(pos_logits)
            #print(neg_logits) # check pos_logits > 0, neg_logits < 0
            optimizer.zero_grad()
            indices = torch.where(pos != 0)
            loss = criterion(pos_logits[indices], pos_labels[indices])
            loss += criterion(neg_logits[indices], neg_labels[indices])
            for param in model.parameters(): loss += config.l2_emb * torch.norm(param)
            loss.backward()
            optimizer.step()
            print("loss in epoch {} iteration {}: {}".format(epoch+1, step, loss.item())) # expected 0.4~0.6 after init few epochs
            wandb.log({"Training Loss by iteration": loss.item()})
            epoch_loss += loss.item()
            
            #raise('stop')
        
        wandb.log({"Training Loss by Epoch": epoch_loss,
                   "Epochs":epoch+1})
        
        if (epoch+1) % 10 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating is_validation:',config.is_validation, end='')
            t_test = evaluation(model, dataset, config.maxlen, device)
            #t_test = evaluation_with_label(model, dataset, config.maxlen, device)
            print('epoch:%d, time: %f(s), (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch+1, T, t_test[0], t_test[1]))
            wandb.log({"NDCG@10": t_test[0],
                       "HT@10":t_test[1]})
            metricsbyepoch[epoch+1] = {'NDCG@10':t_test[0],'HT@10':t_test[1]}
            
            t0 = time.time()
              
            model.train()
    
    return metricsbyepoch

def checkSRFRNudgeUserEmbedding(maxlen,model,device):
    model.eval()
    fakeness_embed = model.embedding_layer.get_fakeness_embed()
    ids = np.array(range(0,maxlen*2+1))
    ids = torch.LongTensor(ids).to(device)
    X = fakeness_embed(ids).cpu().detach().numpy()
    
    linked = linkage(X, 'single')
    labelList = range(0, maxlen*2+1)

    plt.figure(figsize=(10, 7))
    dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
    plt.show()    
    
    
    count = []
    fake_label = getUserLabelNudge(dataset,config.maxlen)
    for i in range(0,maxlen*2+1):
        count.append( np.count_nonzero(fake_label== i - maxlen))
    
    count = np.stack(count)

    topids = count.argsort()[::-1][:10]
    
    
    topids = torch.from_numpy(topids.copy()).to(device)
    Y = fakeness_embed(topids).cpu().detach().numpy()
    
    linked2 = linkage(Y, 'single')
    
    plt.figure(figsize=(10, 7))
    dendrogram(linked2,
            orientation='top',
            labels=topids.tolist(),
            distance_sort='descending',
            show_leaf_counts=True)
    plt.show()  


if __name__ == '__main__':
    wandb.init(project="SRFR Multiple 6 Model Train,Test")
    
    #device = 'cuda' if cuda.is_available() else 'cpu'
    device = 'cpu'
    
    config = wandb.config
    config.inference_only = False
    config.train_batch_size = 128
    config.lr = 0.001 #default 0.001
    config.l2_emb = 0.0
    config.maxlen = 50
    config.num_blocks = 2
    config.num_epochs = 100
    config.num_heads = 1
    config.dropout_rate = 0.5
    config.item_embed_size = 45
    config.fake_embed_size = 5
    config.SEED = 42
    #config.dataset_path = 'data/beauty_discriminated.csv'
    config.dataset_path = 'data/Toys_and_Games.csv'
    #config.dataset_path = 'data/Sports_and_Outdoors.csv'
    #config.dataset_path = 'data/beauty.csv'
    config.is_validation = False
    
    print('running on',device);
    print(config)
    
    torch.backends.cudnn.deterministic = True
    
    # global dataset
    
    df_dataset = pd.read_csv(config.dataset_path)
    print(df_dataset.head(5))
    
    dataset = df_data_partition(df_dataset, is_valid = config.is_validation)

    [user_train, user_test, usernum, itemnum] = dataset
    num_batch = len(user_train['item_ids']) // config.train_batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    
    
    for u in user_train['item_ids']:
        cc += len(user_train['item_ids'][u])
    print('average sequence length: %.2f' % (cc / len(user_train['item_ids'])))
    print(f'user_number : {usernum}, item_number : {itemnum}')
    print(f'number of reviews : {len(df_dataset)}')
    #getUserLabelNudge(dataset,config.maxlen)
    sampler = WarpSampler_fr(user_train, usernum, itemnum, batch_size=config.train_batch_size, maxlen=config.maxlen, n_workers=3)

    model_list = [
        SASRec(
            itemnum,
            config.maxlen,
            config.item_embed_size,
            config.dropout_rate,
            config.num_blocks,
            config.num_heads,
            device
        ).to(device),
        SRFU_B(
            itemnum,
            config.maxlen,
            config.item_embed_size,
            3,
            config.dropout_rate,
            config.num_blocks,
            config.num_heads,
            device
        ).to(device),
        
        SRFU_F(
            itemnum,
            config.maxlen,
            config.item_embed_size,
            config.maxlen+1,
            config.dropout_rate,
            config.num_blocks,
            config.num_heads,
            device
        ).to(device),
        
        SRFU_R(
            itemnum,
            config.maxlen,
            config.item_embed_size,
            11,
            config.dropout_rate,
            config.num_blocks,
            config.num_heads,
            device
        ).to(device)
    ]
    """
    model_list = [
        SASRec(
            itemnum,
            config.maxlen,
            config.item_embed_size,
            config.dropout_rate,
            config.num_blocks,
            config.num_heads,
            device
        ).to(device),
        
        SRFR(
            itemnum,
            config.maxlen,
            config.item_embed_size,
            config.fake_embed_size,
            config.dropout_rate,
            config.num_blocks,
            config.num_heads,
            device
        ).to(device),
        
        SRFRN(
            itemnum,
            config.maxlen,
            config.item_embed_size,
            config.fake_embed_size,
            config.dropout_rate,
            config.num_blocks,
            config.num_heads,
            device
        ).to(device),
        
        SRFU_B(
            itemnum,
            config.maxlen,
            config.item_embed_size,
            3,
            config.dropout_rate,
            config.num_blocks,
            config.num_heads,
            device
        ).to(device),
        
        SRFU_F(
            itemnum,
            config.maxlen,
            config.item_embed_size,
            config.maxlen+1,
            config.dropout_rate,
            config.num_blocks,
            config.num_heads,
            device
        ).to(device),
        
        SRFU_R(
            itemnum,
            config.maxlen,
            config.item_embed_size,
            11,
            config.dropout_rate,
            config.num_blocks,
            config.num_heads,
            device
        ).to(device)
    ]"""
    """
    model_list = [
        SRFR(
            itemnum,
            config.maxlen,
            config.item_embed_size,
            config.fake_embed_size-1,
            config.dropout_rate,
            config.num_blocks,
            config.num_heads,
            device
        ).to(device),
        
        SRFR(
            itemnum,
            config.maxlen,
            config.item_embed_size,
            config.fake_embed_size-2,
            config.dropout_rate,
            config.num_blocks,
            config.num_heads,
            device
        ).to(device),
        SRFR(
            itemnum,
            config.maxlen,
            config.item_embed_size,
            config.fake_embed_size-3,
            config.dropout_rate,
            config.num_blocks,
            config.num_heads,
            device
        ).to(device),
        SRFR(
            itemnum,
            config.maxlen,
            config.item_embed_size,
            config.fake_embed_size-4,
            config.dropout_rate,
            config.num_blocks,
            config.num_heads,
            device
        ).to(device),
        
        SRFRN(
            itemnum,
            config.maxlen,
            config.item_embed_size,
            config.fake_embed_size-1,
            config.dropout_rate,
            config.num_blocks,
            config.num_heads,
            device
        ).to(device),
        SRFRN(
            itemnum,
            config.maxlen,
            config.item_embed_size,
            config.fake_embed_size-2,
            config.dropout_rate,
            config.num_blocks,
            config.num_heads,
            device
        ).to(device),
        SRFRN(
            itemnum,
            config.maxlen,
            config.item_embed_size,
            config.fake_embed_size-3,
            config.dropout_rate,
            config.num_blocks,
            config.num_heads,
            device
        ).to(device),
        SRFRN(
            itemnum,
            config.maxlen,
            config.item_embed_size,
            config.fake_embed_size-4,
            config.dropout_rate,
            config.num_blocks,
            config.num_heads,
            device
        ).to(device),
        

    ]"""
   
    optimizer_list = []
    

    for single_model in model_list:
        for name, param in single_model.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass # just ignore those failed init layers
    
    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)
    
    """
    if config.inference_only:
        for single_model in model_list:
            print('inferenceing')
            single_model.load_state_dict(torch.load('model/SRFR.pt'))
            single_model.eval()
            t_test = evaluation(model, dataset, config.max_len, device)
            print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
            checkSRFRNudgeUserEmbedding(config.maxlen,model,device)
    """ 
    
    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    
    for idx, single_model in enumerate(model_list):
        optimizer_list.append(torch.optim.Adam(single_model.parameters(), lr=config.lr, betas=(0.9, 0.98)))
                
    resultByEpoch = dict()
    resultforExport = []
    
    print(f'simulating on {len(model_list)} model(s)')
    for model_index, single_model in enumerate(model_list):
        print(f'model number {model_index}')
        resultByEpoch[model_index] = simulate(single_model,optimizer_list[model_index],bce_criterion,sampler,config,wandb,model_index)
        
        pd.DataFrame.from_dict(resultByEpoch[model_index],orient='index',columns=['NDCG@10','HT@10']).to_csv('result/result_model_'+str(model_index)+'.csv')
        
        resultforExport = evaluation_with_label(single_model, dataset, config.maxlen, device)
        
        pd.DataFrame.from_dict(resultforExport[2],orient='index',columns=['Rank','NDCG@10','HT@10','binary_label','frequency_label','ratio_label']).to_csv('result/result_user_model_'+str(model_index)+'.csv')
        pd.DataFrame.from_dict(resultforExport[3],orient='index',columns=['NDCG@10','HT@10','number']).to_csv('result/result_BL_model_'+str(model_index)+'.csv')
        pd.DataFrame.from_dict(resultforExport[4],orient='index',columns=['NDCG@10','HT@10','number']).to_csv('result/result_FL_model_'+str(model_index)+'.csv')
        pd.DataFrame.from_dict(resultforExport[5],orient='index',columns=['NDCG@10','HT@10','number']).to_csv('result/result_RL_model_'+str(model_index)+'.csv')
        
        if ~config.inference_only :
            torch.save(single_model.state_dict(), 'model/SRFR_'+str(model_index)+'.pt')
            print("Exporting Model parameters at 'model/SRFR_"+str(model_index)+".pt'")
    
    sampler.close()
    
    print(resultByEpoch)
    print("Done")
    
    
    """
    T = 0.0
    t0 = time.time()
    
    for epoch in range(config.num_epochs):
        if config.inference_only: break
        
        epoch_loss = 0
        
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, rsq, pos, prs, neg, nrs = sampler.next_batch() # tuples to ndarray
            u, seq, rsq, pos, prs, neg, nrs = np.array(u), np.array(seq), np.array(rsq), np.array(pos), np.array(prs), np.array(neg) , np.array(nrs)
            u, seq, rsq, pos, prs, neg, nrs = torch.LongTensor(u).to(device), torch.LongTensor(seq).to(device), torch.LongTensor(rsq).to(device), torch.LongTensor(pos).to(device), torch.LongTensor(prs).to(device), torch.LongTensor(neg).to(device), torch.LongTensor(nrs).to(device),
            #print(u.shape,seq.shape,rsq.shape,pos.shape,neg.shape)
            hidden_state, pos_logits, neg_logits = model(user_ids = u , input_ids = seq, fake_ids = rsq, positive_ids = pos, positive_fake_ids=prs, negative_ids = neg, negative_fake_ids=nrs)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=device), torch.zeros(neg_logits.shape, device=device)
            #print("\neye ball check raw_logits:")
            #print(pos_logits)
            #print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = torch.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.parameters(): loss += config.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            print("loss in epoch {} iteration {}: {}".format(epoch+1, step, loss.item())) # expected 0.4~0.6 after init few epochs
            wandb.log({"Training Loss by iteration": loss.item()})
            epoch_loss += loss.item()
            
            #raise('stop')
        
        wandb.log({"Training Loss by Epoch": epoch_loss,
                   "Epochs":epoch+1})
        
        if (epoch+1) % 1 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating is_validation:',config.is_validation, end='')
            t_test = evaluation(model, dataset, config.maxlen, device)
            print('epoch:%d, time: %f(s), (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, t_test[0], t_test[1]))
            wandb.log({"NDCG@10": t_test[0],
                       "HT@10":t_test[1]})
            t0 = time.time()
            
            
            model.train()
    """
        #if epoch == config.num_epochs - 1:
        #    if not config.inference_only :
        #        torch.save(model.state_dict(), 'model/SRFR.pt')
        #        print("Exporting Model parameters at 'model/SRFR.pt'")
        #    checkSRFRNudgeUserEmbedding(config.maxlen,model,device)
    
    #checkSRFRNudgeUserEmbedding(config.maxlen,model,device)
    """
    temp0, temp1, userLabelviaResult, userResults = evaluation_nudge(model, dataset, config.maxlen, device)
    
    wandb.log({"final_NDCG@10":temp0,
                "final_HT@10":temp1})
    
    pd.DataFrame.from_dict(userLabelviaResult,orient='index',columns=['HT@10','NDCG@10','user_cnt']).to_csv('resultvialabel.csv')
    pd.DataFrame.from_dict(userResults,orient='index',columns=['rank', 'user_HIT@10', 'user_NDCG@10', 'user_real_reviews', 'user_fake_reviews']).to_csv('userResults.csv')
    
    if ~config.inference_only :
        torch.save(model.state_dict(), 'model/SRFR_SASRec.pt')
        print("Exporting Model parameters at 'model/SRFR_SASRec.pt'")
    """
    #sampler.close()
    #print("Done")
