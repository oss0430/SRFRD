import sys
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
from torch import cuda
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import wandb

"""
Possible Discriminator Train Dataset

Amazon_Reviews
https://www.kaggle.com/datasets/lievgarcia/amazon-reviews

currently ... precision 0.691 recall 0.641 f1 0.665 acc 0.678
at inference all positive or all negative currently all positive(1)

final 2 : pre 0.689 rec 0.604 f1 0.644 acc 0.666
fake - 6795 real - 720
"""
labels = {
          '__label1__' : 0, 
          '__label2__' : 1
          }
#__label1__ is fake
#__label2__ is not fake          
save_path = './saved_model'
saved_path  = './saved_model/'

device = 'cuda' if cuda.is_available() else 'cpu'
print(device)
torch.cuda.empty_cache() 
epsilon = sys.float_info.epsilon
class FakeReviewDataset(Dataset):

    def __init__(self, dataframe, tokenizer, input_max_len):
        self.tokenizer  = tokenizer 
        self.data       = dataframe
        self.input_text = self.data['REVIEW_TEXT']
        self.labels     = self.data['LABEL']
        self.input_max_len = input_max_len
        
    def classes(self):
        return self.labels
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        input_text = str(self.input_text[index])
        input_text = ' '.join(input_text.split())
        input_text = self.tokenizer([input_text],
                                    padding = 'max_length',
                                    max_length=self.input_max_len,
                                    truncation = True,
                                    return_tensors="pt")

        input_text_ids = input_text['input_ids'].squeeze()
        input_mask     = input_text['attention_mask']
        
        labels_y       = self.labels[index]
        labels_y       = labels[labels_y]
        labels_y       = torch.tensor([labels_y])
        
        return {
            'input_text_ids' : input_text_ids.to(dtype=torch.long),
            'input_mask'     : input_mask.to(dtype=torch.long),
            'labels_y'       : labels_y.to(dtype=torch.long)
        }
        

def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    total_correct = 0
    total_len     = 0
    
    for _,data in enumerate(loader, 0):
        optimizer.zero_grad()
        
        y    = data['labels_y'].to(device, dtype = torch.long)
        ids  = data['input_text_ids'].to(device, dtype = torch.long)
        mask = data['input_mask'].to(device, dtype = torch.long)
        
        #print(y)
        outputs = model(input_ids = ids, attention_mask = mask, labels=y)
        
        loss = outputs[0]
        logits = outputs.logits
        pred  = torch.argmax(F.softmax(logits,dim=1),dim=1)
        
        correct = pred.eq(y)
        total_correct += correct.sum().item()
        total_len += len(labels)
        loss.backward()
        optimizer.step()
        
        if _%10 == 0:
            wandb.log({"Training Loss": loss.item()})
        
        if _%500==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
    
    print(f'Epoch: {epoch}, Accuracy: {total_correct/total_len:.3f}')

def validate(epoch, tokenizer, model, device, loader):
    model.eval()

    total_correct_test = 0
    
    tp = 0 
    fp = 0
    tn = 0
    fn = 0
    
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y    = data['labels_y'].to(device, dtype = torch.long)
            ids  = data['input_text_ids'].to(device, dtype = torch.long)
            mask = data['input_mask'].to(device, dtype = torch.long)
  
            logits = model(input_ids = ids, attention_mask = mask).logits
            predicted_class_id = torch.argmax(F.softmax(logits,dim=1), dim=1)
            
            #print(predicted_class_id.item())
            
            if predicted_class_id.item() == 1 :
                if predicted_class_id.item() == y.item() :
                    tp += 1
                else:
                    fp += 1
            else :
                if predicted_class_id.item() == y.item() :
                    tn += 1
                else:
                    fn += 1 
            
            #correct = torch.sum(predicted_class_id == y)
            #print(y, predicted_class_id, torch.sum(predicted_class_id == y))
            
            
            #acc = (predicted_class_id == y).sum()
            #total_correct_test += correct
            
            if _%100==0:
                print(f'Completed {_}')
    
    precision = tp/(tp+fp+epsilon) + epsilon
    recall    = tp/(tp+fn+epsilon) + epsilon 
    f1        = 2/(1/precision+1/recall)
    acc       = (tp+tn)/len(loader)

    return precision, recall, f1, acc
    
def main():
    wandb.init(project="BERT_Fake_Review_Detection results")
    
    config = wandb.config           # Initialize config
    config.TRAIN_BATCH_SIZE = 32    # input batch size for training (default: 64)
    config.VALID_BATCH_SIZE = 1     # input batch size for testing (default: 1)
    config.TRAIN_EPOCHS =  10       # number of epochs to train (default: 10)
    config.VAL_EPOCHS = 1  
    config.LEARNING_RATE = 4.00e-05 # learning rate (default: 0.01)
    config.SEED = 420               # random seed (default: 42)
    config.MAX_LEN = 512
    
    
    
    train_params = {
        'batch_size': config.TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
        }

    val_params = {
        'batch_size': config.VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
        }
    
    torch.backends.cudnn.deterministic = True
    
    
    dfdataset  = pd.read_csv('fakeReviewDetectionDataset/amazon_reviews.txt', delimiter='\t')
    dftrainset = dfdataset.sample(frac=0.8,random_state=config.SEED)
    dftestset  = dfdataset.drop(dftrainset.index)
    dftrainset.reset_index(drop=True, inplace=True)
    dftestset.reset_index(drop=True, inplace=True)
    
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    training_set = FakeReviewDataset(dftrainset, tokenizer, config.MAX_LEN)
    test_set     = FakeReviewDataset(dftestset , tokenizer, config.MAX_LEN)
    
    train_loader = DataLoader(training_set, **train_params)
    test_loader  = DataLoader(test_set, **val_params)
    
    print(dftrainset.sample(10))
    print("TRAIN Dataset: {}".format(dftrainset.shape))

    
    
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    model.to(device)
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=config.LEARNING_RATE)
    #optimizer = AdamW(params=model.parameters(), lr=config.LEARNING_RATE, eps=epsilon, correct_bias=False) 
    
    print("Training")
    for epoch in range(config.TRAIN_EPOCHS):
        train(epoch, tokenizer, model, device, train_loader, optimizer)
    
    print("saving parameter")
    model.save_pretrained(save_path)
    
    """
    #model = BertForSequenceClassification.from_pretrained(saved_path)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    model.to(device)
    """
    
    print("Validating")
    
    for epoch in range(config.VAL_EPOCHS):
        precision, recall, f1, acc = validate(epoch, tokenizer, model, device, test_loader)
        print(f"precision:{precision:.3f}\nrecall:{recall:.3f}\nf1:{f1:.3f}\nacc:{acc:.3f}")
        


    
if __name__ == '__main__':
    main()