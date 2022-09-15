import sys
import pandas as pd

class fakeReviewDatasetParser():
    def __init__(self, path, review_column_name, label_column_name, fake_label):
        self.dataset = open(path)
        self.review_column_name = review_column_name
        self.label_column_name = label_column_name
        self.fake_label = fake_label
        
    
    def _process_line(self, line):
        return
        
    def to_csv(self):
        return


class amazonFakeReviewDatasetParser(fakeReviewDatasetParser):
    def _process_line(self,line):
        print(line.split())
    
    def to_csv(self,path):
        lines = self.dataset.readlines()
        keys = lines[0].split()
        
        reviewIDX = -1
        labelIDX = -1
        
        for idx, keyName in enumerate(lines[0].split()):
            if keyName == self.review_column_name :
                reviewIDX = idx
            if keyName == self.label_column_name :
                labelIDX  =  idx 
    
        datasetDict = {'label':[], 'review':[]}
        
        for line in lines[1:]:
            row = line.split(maxsplit = 8)
            print(row)
            if row[labelIDX] == self.fake_label :
                datasetDict['label'].append('fake')
            else :    
                datasetDict['label'].append('real')
            datasetDict['review'].append(row[reviewIDX])
        
        return pd.DataFrame(datasetDict).to_csv(path)
        


testDataset = amazonFakeReviewDatasetParser('fakeReviewDetectionDataset/amazon_reviews.txt','REVIEW_TEXT','LABEL','__label1__')
#testDataset.to_csv('fakeReview.csv')

print(pd.read_csv('fakeReviewDetectionDataset/amazon_reviews.txt', delimiter='\t').sample(5))