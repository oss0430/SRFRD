import os
import sys
import random
import gzip
import json
import pandas as pd
import numpy as np
import tqdm
import time

from collections import defaultdict
from datetime import datetime

import pandas as pd
from tqdm import tqdm



print(sys.executable)
os.chdir('/home/iknow/Desktop/SeqWithFakeDetection/SRFRD/data/raw/yelp')


# for eval()
true = True
false = False

yelp_column_map = {'user_id':'user_id','item_id':'item_id','time':'time','review':'review','friend_user_id':'user_id','friend_friend_list':'friend_list'}


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
    
    #max_index = 1000000
    parsed_dict = {}
    #review_dataframe = pd.DataFrame(columns=['user_id','item_id','date','text'])
    i = 0
    with open(path) as fin:
        for line in tqdm(fin):
        
            #if i == max_index :
            #   break
                
                
            line_contents = json.loads(line)
            parsed_dict[i] = {'user_id':line_contents['user_id'], 'item_id' : line_contents['business_id'], 'time' : line_contents['date'], 'review' : line_contents['text']}
            i += 1
            
            #if(i%1000 == 0): print(i)
            #review_dataframe = pd.concat([review_dataframe, pd.DataFrame({'user_id':[line_contents['user_id']], 'item_id' : [line_contents['business_id']], 'time' : [line_contents['date']], 'review' : [line_contents['text']]})])
    
    #review_datafrmae
    
    return pd.DataFrame.from_dict(parsed_dict, orient = 'index')

def text_to_list(text):
    #
    return text.split(', ')
    
def yelp_load_friend_getDF(path):
    
    #max_index = 10000
    
    parsed_dict = {}
    #user_dataframe = pd.DataFrame(columns=['user_id','friend_list'])
    i = 0
    with open(path) as fin:
        for line in tqdm(fin):
            #if i == max_index :
            #    break
            
            line_contents = json.loads(line)
            friend_list = text_to_list(line_contents['friends'])
            parsed_dict[i] = {'user_id':line_contents['user_id'], 'friend_list' : friend_list}
            
            i += 1
            #if(i%1000 == 0): print(i)
            
            #user_dataframe = pd.concat([user_dataframe, pd.DataFrame({'user_id':[line_contents['user_id']], 'friend_list' : [friend_list]})])
            
    return pd.DataFrame.from_dict(parsed_dict, orient = 'index')

def dfyelp_parse_friend_getDF(df):
    """
    when friend_list section is not list just text
    """
    user_id_column_name = "user_id"
    friend_list_column_name = "firend_list"
    friend_dict = {}
    
    for idx, row in df.iterrows():
        friend_list = text_to_list(row[friend_list_column_name])
        friend_dict[idx] = {'user_id':row[user_id_column_name],'friend_list':friend_list}
        if idx%1000 == 0:
            print(idx)
        
    return pd.DataFrame.from_dict(friend_dict, orient = 'index')

def count_elements(df, column_map):
    
    countU = defaultdict(lambda: 0)
    countI = defaultdict(lambda: 0)
    
    user_id_column_name = column_map['user_id']
    item_id_column_name = column_map['item_id']
    
    for idx, row in tqdm(df.iterrows(),total=len(df)):
        item_raw_id = row[item_id_column_name]
        user_raw_id = row[user_id_column_name]
        
        countU[user_raw_id] += 1
        countI[item_raw_id] += 1
        
        #if idx%1000 == 0 :    print(idx)
        
    return countU, countI


def clean_friend(userMapDict, dfFriend, column_map):
    
    dfFriendClean = pd.DataFrame(columns=['user_id','friend_list'])
    
    friendClean = []
    friend_user_id_column_name = column_map['friend_user_id']
    friend_friend_list_column_name = column_map['friend_friend_list']
    
    for idx, row in tqdm(dfFriend.iterrows(),total=len(dfFriend)):
        user_raw_id = row[friend_user_id_column_name][0]
        friend_list = row[friend_friend_list_column_name]
        userid = ''
        if user_raw_id not in userMapDict: #don't exist in the Map
            pass
            
        else:   
            userid = userMapDict[user_raw_id]
        
        clean_friend_list = []
        
        for friend in friend_list :
            if friend not in userMapDict: #don't exist in the Map
                pass
                
            else :
                clean_friend_list.append(userMapDict[friend][0])
        
        if len(clean_friend_list) > 0 :
            friendClean.append([userid,clean_friend_list])
    
    #print(friendClean)
                
    return pd.DataFrame(friendClean,columns=['user_id','friend_list'])

def clean_review(df, column_map):
    
    dfUserMap = pd.DataFrame(columns=['user_id'])
    userMapDict = {}
    
    dfItemMap = pd.DataFrame(columns=['item_id'])
    itemMapDict = {}
    
    dfReviewClean = pd.DataFrame(columns=['user_id','time','item_id','review'])
    
    user_id_column_name = column_map['user_id']
    item_id_column_name = column_map['item_id']
    review_column_name  = column_map['review']
    time_column_name    = column_map['time']
    
    print("counting")
    countU, countI = count_elements(df, column_map)
    print("counting complete")
    
    userid = 0
    itemid = 0
    usernum = 0
    itemnum = 0
    
    print("cleaning review")
    clean_review_list = []
    #user_map_list = []
    #item_map_list = []
    #
    #
    
    for idx, row in tqdm(df.iterrows(),total=len(df)):
    
        user_raw_id = row[user_id_column_name]
        item_raw_id = row[item_id_column_name]
        date        = row[time_column_name]
        review      = row[review_column_name]
        

        if countU[user_raw_id] < 5 or countI[item_raw_id] < 5: #discard this interaction
            continue

        if user_raw_id in userMapDict:
            user_id = userMapDict[user_raw_id]
        
        else:
            usernum += 1
            userid = usernum
            userMapDict[user_raw_id] = usernum

        if item_raw_id in itemMapDict:
            itemid = itemMapDict[item_raw_id]
        
        else:
            itemnum += 1
            itemid = itemnum
            itemMapDict[item_raw_id] = itemid
        

        clean_review_list.append([userid,date,itemid,review])

    
    dfReviewClean = pd.DataFrame(clean_review_list, columns=['user_id','time','item_id','review'])
    print("cleaning complete now sorting")    
    dfReviewClean.sort_values(by=['user_id','time'], inplace=True)   
    
    return userMapDict, itemMapDict, dfReviewClean
    #return dfUserMap, dfItemMap, dfReviewClean
    
def main():
    """
    PARAMETERS
    """

    use_review = False
    dataset_name = r'Yelp'
    dataset_path = "/home/iknow/Desktop/SeqWithFakeDetection/SRFRD/data/raw/yelp/yelp_academic_dataset_review.json"
    dataset_csv_path = ""
    
    #"/home/iknow/Desktop/SeqWithFakeDetection/SRFRD/data/raw/yelp/yelp_academic_dataset_user.json"
    isAmazon = False #if False is Yelp
    start_from_csv = False #if False load from gzip or json
    
    save_user_map = False
    load_user_map = True #if False
    user_map_path = "/home/iknow/Desktop/SeqWithFakeDetection/SRFRD/data/raw/yelp/UserMap_Yelp.csv"
     
    use_friend  = True
    friend_dataset_path = "/home/iknow/Desktop/SeqWithFakeDetection/SRFRD/data/raw/yelp/yelp_academic_dataset_user.json"
    friend_csv_path = "/home/iknow/Desktop/SeqWithFakeDetection/SRFRD/data/raw/yelp/user_json.csv"
    
    column_map  = yelp_column_map    
    
    """
    Functionalities :
    
    clean review to better visible id value
    
    discard user and items with so little interactions
    
    clean friend list to better visible id value
    """
    
    print("Loading Start")
    
    if not start_from_csv :
        if isAmazon:
            if use_review:
                dfReview = amazon_load_getDF(dataset_path)
        else :
            if use_review:
                dfReview = yelp_load_getDF(dataset_path)
                print(dfReview.head(5))
            if use_friend :
                dfFriend = yelp_load_friend_getDF(friend_dataset_path)
                print(dfFriend.head(5))    
        
            
    else :
        if use_review:
            dfReview = pd.load_csv(dataset_csv_path)
        
        if use_friend :
            dfFriend = dfyelp_parse_friend_getDF(pd.load_csv(friend_csv_path))
            
    print("Loading Complete")
    
    if use_review:
        print("Cleaning Review")    
        #dfUserMap, dfItemMap, dfReviewClean = clean_review(dfReview, column_map)
        userMapDict, itemMapDict, dfReviewClean = clean_review(dfReview, column_map)
        
        print("Cleaning Review Complete")
        print(f"user # : {len(userMapDict)} | item # : {len(itemMapDict)}")
        print(f"total interactions {len(dfReviewClean)}")
        print(dfReviewClean.head(10))
    
    if load_user_map:
        print("loading User Map")
        dfUserMap = pd.read_csv(user_map_path, index_col = 0)
        print(dfUserMap.head(10))
        userMapDict = dfUserMap.T.to_dict('list')
        
        #print(userMapDict)
        
    if use_friend :
        print("Cleaning Friend")
        dfFriendClean = clean_friend(userMapDict, dfFriend, column_map)
        print("Cleaning Friend Complete")
        print(dfFriendClean.head(10))
    
    if use_review:
        print("Exporting Review")  
        dfReviewClean.to_csv("ReviewClean_"+dataset_name+".csv")
        print("Exporting Review Complete")
    
    if save_user_map :
        print("Exporting User Map")
        pd.DataFrame.from_dict(userMapDict,orient = 'index',columns=['user_raw_id','userid']).to_csv("UserMap_" + dataset_name + ".csv")
        print("Exporting User Map Complete")
        
    if use_friend :
        print("Exporting Friend") 
        dfFriendClean.to_csv("FriendClean_"+dataset_name+".csv")
        print(f"total friend list {len(dfFriendClean)}")
        print("Exporting Friend Complete")

if __name__ == '__main__':
    main()