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


"""
This is for the yelp DATASET

We will measure the user representation level with Yelp Friends list just like

 "Xu Xie, et al. Constrastive Learning for Sequential Recommendation. 2022. IEE 38th ICDE"

"""

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


    

def checkUserRepresentationViaLabel(
    data,
    friend_dict,
    number_of_labels,
    labeling_function,
    device):
    
    """
    for each label we check,
    
    number of friends in the Yelp dataset
    
    if inside the label check if he/she is friend if they are add to count
    """
    
    number_of_friend_relations = defaultdict()
    
    
    MAE_scaled_dict = group_user_vial_labelMAE(labeling_function,dataset,maxlen)
    
    plt.bar(range(len(MAE_scaled_dict)), list(MAE_scaled_dict.values()), align='center')
    plt.xticks(range(len(MAE_scaled_dict)), list(MAE_scaled_dict.keys()))
    
    plt.show()  
    

def checkUserRepresentationViaCosSim(
    model,
    dataset,
    maxlen,
    friend_dict,
    device)
    
    """
    for each user we check its frineds
    
    measure avg cos_similarity with user and his/her friends
    
    group avg cos_similarity with window size of 0.05
    """
    
    similarity_dict = group_user_via_embeddingSimilarity(model, dataset, maxlen, friend_dict)
    
    plt.bar(range(len(similarity_dict)), list(similarity_dict.values()),align='center')
    plt.xticks(range(len(similarity_dict)), list(similarity_dict.keys()))
    
    plt.show()
    
    return similarity_dict

if __name__ == '__main__':
    wandb.init(project="SRFR Multiple 6 Model Train,Test")
    
    #device = 'cuda' if cuda.is_available() else 'cpu'
    device = 'cuda'
    
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
    config.friend_path = ''
    
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
    
    df_friends = pd.read_csv(config.friend_path)
    print(df_friends.head(5))
    
    friend_dict = df_friends.to_dict()
    
    
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
        
        ## added for friend similarity
        pd.DataFrame.from_dict(checkUserRepresentationViaCosSim(single_model,dataset,config.maxlen,friend_dict,device))
        
        if ~config.inference_only :
            torch.save(single_model.state_dict(), 'model/SRFR_YELP'+str(model_index)+'.pt')
            print("Exporting Model parameters at 'model/SRFR_YELP"+str(model_index)+".pt'")
    
    sampler.close()
    
    print(resultByEpoch)
    print("Done")