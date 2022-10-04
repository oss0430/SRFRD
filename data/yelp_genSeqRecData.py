import os
import sys
import random
import gzip
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime

import pandas as pd
from tqdm import tqdm

print(sys.executable)
os.chdir('/home/iknow/Desktop/SeqWithFakeDetection/SRFRD/data/raw/yelp')


# for eval()
true = True
false = False

yelp_column_map = {'user_id'}



"""
functions
"""

def parse_gzip(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def amazon_load_getDF(path):
    i = 0
    df = {}
    for d in parse_gzip(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient = 'index')

    
def yelp_load_getDF(path):

    review_dataframe = pd.DataFrame(columns=['user_id','item_id','date','text'])

    with open(review_json_path) as fin:
        for line in (fin):
            line_contents = json.loads(line)
            review_dataframe = pd.concat([review_dataframe, pd.DataFrame({'user_id':[line_contents['user_id']], 'item_id' : [line_contents['business_id']], 'date' : [line_contents['date']], 'text' : [line_contents['text']]})])
    
    return review_dataframe

def text_to_list(text):
    #
    return string.split(', ',text)
    
def yelp_load_friend_getDF(path):

    user_dataframe = pd.DataFrame(columns=['user_id','friend_list'])

    with open(user_json_path) as fin:
    for line in tqdm(fin):
        line_contents = json.loads(line)
        friend_list = text_to_list(line_contents['friends'])
        user_dataframe = pd.concat([user_dataframe, pd.DataFrame({'user_id':[line_contents['user_id']], 'friend_list' : [friend_list]})])
            
    return user_dataframe

def count_elements(df, column_map):
    
    countU = defaultdict(lambda: 0)
    countI = defaultdict(lambda: 0)
    
    user_id_column_name = column_map['user_id']
    item_id_column_name = column_map['item_id']
    
    for idx, row in df.iterrows():
        item_raw_id = row[item_id_column_name]
        user_raw_id  = row[user_id_column_name]
        countU[user_raw_id]  += 1
        countI[item_raw_id] += 1
    
    return countU, countI


def clean_friend(dfUserMap,dfFriend,column_map):
    
    dfFriendClean = pd.DataFrame(columns=['user_id','time','item_id','review'])
    
    
    friend_user_id_column_name = column_map['friend_user_id']
    friend_friend_list_column_name = column_map['friend_friend_list']
    
    for idx, row in dfFriend.iterrows():
        user_raw_id = row[friend_user_id_column_name]
        friend_list = row[friend_friend_list_column_name]
        
        if user_raw_id not in dfUserMap.values: #don't exist in the Map
            pass
            
        else:   
            userid = dfUserMap.index[dfUserMap['user_id']==user_raw_id].tolist()[0]
        
        clean_friend_list = []
        
        for friend in friend_list :
            if friend not in dfUserMap.values: #don't exist in the Map
                pass
                
            else :
                clean_friend_list.append(dfUserMap[friend])
        
        newFriendClean = pd.DataFrame.from_dict({'user_id':[userid],'friend_list':[clean_friend_list]})
        dfFriendClean  = pd.concat([dfFriendClean,newFriendClean], ignore_index=True)
        
                
    return dfFriendClean

def clean_review(df, column_map):
    
    dfUserMap = pd.DataFrame(columns=['user_id'])
    dfItemMap = pd.DataFrame(columns=['item_id'])
    dfReviewClean = pd.DataFrame(columns=['user_id','time','item_id','review'])
    
    user_id_column_name = column_map['user_id']
    item_id_column_name = column_map['item_id']
    review_column_name  = column_map['review']
    time_column_name    = column_map['time']
    
    countU, countI = count_elements(df, column_map)
    
    for idx, row in df.iterrows():
    
        user_raw_id = row[user_id_column_name]
        item_raw_id = row[item_id_column_name]
        time        = row[time_column_name]
        review      = row[review_column_name]
        
  
        if countU[user_raw_id] < 5 or countI[item_raw_id] < 5: #discard this interaction
            continue

        if user_raw_id in dfUserMap.values: #already exist in the Map
            userid = dfUserMap.index[dfUserMap['user_id']==user_raw_id].tolist()[0]
   
        else: #non existing User in Map make one
            usernum += 1
            userid = usernum
            newUser = pd.DataFrame({'user_id':[user_raw_id]},index=[userid])
            dfUserMap = pd.concat([dfUserMap,newUser])
        
        if item_raw_id in dfItemMap.values: #already exist in the Map
    
            itemid = dfItemMap.index[dfItemMap['item_id']==item_raw_id].tolist()[0]

        else: #non existing Item in Map make one
            itemnum += 1
            itemid = itemnum
            newItem = pd.DataFrame({'item_id':[asin]},index=[itemid])
            dfItemMap = pd.concat([dfItemMap,newItem])

        newReviewClean = pd.DataFrame.from_dict({'user_id':[userid],'time':[time],'item_id':[itemid],'review':[review]})
        dfReviewClean = pd.concat([dfReviewClean,newReviewClean], ignore_index=True)
        
        
    dfReviewClean.sort_values(by=['user_id','time'], inplace=True)   
    
    return dfUserMap, dfItemMap, dfReviewClean
    
def main():
    """
    PARAMETERS
    """

    dataset_name = r'Yelp'
    dataset_path = r'./raw/yelp/yelp_academic_dataset_review.json'
    
    
    isAmazon = False #if False is Yelp
    start_from_csv = True #if False load from gzip or json
    
    use_friend  = True
    friend_dataset_path = r'./raw/yelp/yelp_academic_dataset_user.json'
    
    
    column_map  = {}
    
    
    """
    Functionalities :
    
    clean review to better visible id value
    
    discard user and items with so little interactions
    
    clean friend list to better visible id value
    """
    
    print("Loading Start")
    
    if not start_from_csv :
        if isAmazon:
            dfReview = amazon_load_getDF(dataset_path)
        else :
            dfReview = yelp_load_getDF(dataset_path)
            
            if use_friend :
                dfFriend = yelp_load_friend_getDF(friend_dataset_path)
            
    else :
        dfReview = pd.load_csv(dataset_path)
        
        if use_friend :
            dfFriend = pd.load_csv(friend_dataset_path)
    
    print("Loading Complete")
        
    print("Cleaning Review")    
    dfUserMap, dfItemMap, dfReviewClean = clean_review(dfReview, columnm_map)
    
    print("Cleaning Review Complete")
    
    if use_friend :
        print("Cleaning Friend")
        dfFriendClean = clean_friend(dfUserMap,dfFriend,column_map)
        print("Cleaning Friend Complete")

    print("Exporting Review")  
    dfReviewClean.to_csv("ReviewClean_"+datasetName+".csv")
    print("Exporting Review Complete")
    
    if use_friend :
        print("Exporting Friend") 
        dfFriendClean.to_csv("FriendClean_"+datasetName+".csv")
        print("Exporting Friend Complete")

if __name__ == '__main__':
    main()