import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
from torch import cuda
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from transformers import BertTokenizer, BertForSequenceClassification

import wandb

dataset_name = 'Yelp'
dataset_path = "/home/iknow/Desktop/SeqWithFakeDetection/SRFRD/data/raw/yelp/review_json.csv"
#Sports_and_Outdoors
#beauty

saved_path  = './saved_model/'
weight_path = './saved_model/pytorch_model.bin'
config_path = './saved_model/config.json'


device = 'cuda' if cuda.is_available() else 'cpu'
print(device)
torch.cuda.empty_cache() 

class AmazonReviewDataset(Dataset):

    def __init__(self, dataframe, tokenizer, input_max_len):
        self.tokenizer  = tokenizer 
        self.data       = dataframe
        self.input_text = self.data['text'] ## originally ['review']
        self.input_max_len = input_max_len
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_text = str(self.input_text[index])
        input_text = ' '.join(input_text.split())
        input_text = self.tokenizer([input_text],
                                    padding = 'max_length',
                                    max_length=self.input_max_len,
                                    truncation = True,
                                    return_tensors="pt")

        input_text_ids = input_text['input_ids'].squeeze()
        input_mask     = input_text['attention_mask'].squeeze()
        

        
        return {
            'input_text_ids': input_text_ids.to(dtype=torch.long),
            'input_mask'    : input_mask.to(dtype=torch.long),
        }

def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    
    generated_labels = []
    
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            ids  = data['input_text_ids'].to(device, dtype = torch.long)
            mask = data['input_mask'].to(device, dtype = torch.long)
  
            logits = model(ids).logits
            pred  = torch.argmax(F.softmax(logits,dim=1),dim=1)
            
            generated_labels.extend([pred])
            
            if _%100==0:
                print(f'Completed {_}')

    return generated_labels
    
    
def main():
    wandb.init(project="BERT_Fake_Review_Detection using results")
    
    config = wandb.config           # Initialize config
    config.VALID_BATCH_SIZE = 1     # input batch size for testing (default: 1000)
    config.VAL_EPOCHS = 1  
    config.SEED = 42                # random seed (default: 42)
    config.MAX_LEN = 512
    
    val_params = {
        'batch_size': config.VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
        }
    
    torch.backends.cudnn.deterministic = True
    
    print('loading dataset ',dataset_name)
    
    #dfdataset  = pd.read_csv('AmazonReviewDataset/'+dataset_name+'.csv')
    #ReviewCleanreviews_Toys_and_Games.csv
    #ReviewCleanreviews_Sports_and_Outdoors.csv
    dfdataset = pd.read_csv(dataset_path)
    print('Discriminating ',len(dfdataset),' reviews') 
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    dataset = AmazonReviewDataset(dfdataset, tokenizer, config.MAX_LEN)
    
    test_loader = DataLoader(dataset, **val_params)
    
    model = BertForSequenceClassification.from_pretrained(saved_path)
    model.to(device)

    print("Validating")
    
    generated_label = []
    for epoch in range(config.VAL_EPOCHS):
        generated_label = validate(epoch, tokenizer, model, device, test_loader)
    
    fake_review  =  []
    for idx, label in enumerate(generated_label):
        if label == 0:
            fake_review.append('fake')
        else :
            fake_review.append('real')
        
    dfdataset['fake_review'] = fake_review
    
    """
    #rule 1 if most fake review is a fake person
    user_fake_label_rule1 = []
    user_id  = -1
    user_cnt = 0
    user_fake_review = 0
    user_is_fake = 'real'
    for idx,row in dfdataset.iterrows():
        if user_id is not row['user_id'] :
            if user_fake_review > user_cnt/2 :
                user_is_fake = 'fake'
            else :
                user_is_fake = 'real'
            for i in range(user_cnt):
                user_fake_label_rule1.append(user_is_fake)
            
            user_id  = row['user_id']
            user_cnt = 0
            user_fake_review = 0
            user_is_fake = 'real'
            
        user_cnt += 1
        if row['fake_review'] == 'fake' :
            user_fake_review += 1
    

    if user_fake_review > user_cnt/2 :
        user_is_fake = 'fake'
    else :
        user_is_fake = 'real'
    for i in range(user_cnt):
        user_fake_label_rule1.append(user_is_fake)

    
    #rule 2 if one real review is a real person
    user_fake_label_rule2 = []
    user_id  = -1
    user_cnt = 0
    user_fake_review = 0
    user_is_fake = 'real'
    for idx,row in dfdataset.iterrows():
        if user_id is not row['user_id'] :
            if user_fake_review < user_cnt :
                user_is_fake = 'real'
            else :
                user_is_fake = 'fake'
            for i in range(user_cnt):
                user_fake_label_rule2.append(user_is_fake)
            
            user_id  = row['user_id']
            user_cnt = 0
            user_fake_review = 0
            user_is_fake = 'real'
            
        user_cnt += 1
        if row['fake_review'] == 'fake' :
            user_fake_review += 1
    

    if user_fake_review < user_cnt :
        user_is_fake = 'real'
    else :
        user_is_fake = 'fake'
    for i in range(user_cnt):
        user_fake_label_rule2.append(user_is_fake)
        
    #dfdataset['user_fake_label_rule1'] = user_fake_label_rule1
    #dfdataset['user_fake_label_rule2'] = user_fake_label_rule2
    """
    dfdataset.to_csv(dataset_name+'_final_data.csv')
    
    
    
    print('Generated File')

    
if __name__ == '__main__':
    main()