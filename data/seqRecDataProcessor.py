import os
import sys
import random
import gzip
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
 
# for eval()
true = True
false = False

datasetName = r'All_Beauty'

"""
dataset name
"""
dataset_review = r'raw/'+datasetName+r'.json.gz' 

"""
load data
"""
print("Loading Review {}".format(datasetName))
#Load from gzip
def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient = 'index')

dfReview = getDF(dataset_review) ##this is where all the interactions are at
dffakeReview = pd.read_csv('fakeReviewResult/'+datasetName+'.csv') ##this is the fake/non-fake predicted labels

print("Loading Complete")

"""
label processing :
  - count user's fake review and non-fake review, if fake review outnumbers non-fake review the user is considered fake
"""
for idx, row in dffakeReview.iterrows():
    print(row)

"""
review processing :
  - Count User and Item, countU = {user: count}  countI = {item: counmt}
  - Discard User and Item which has less interactions count than 5
  - Map User and Item into more reasonable ID (4 to 6 figures)
  - Clean Meta Data and match Item ID with Mapped ID
"""
#Count user and item
countU = defaultdict(lambda: 0)
countI = defaultdict(lambda: 0)

print("Counting Reviews")

for idx, row in dfReview.iterrows():
    asin = row['asin']
    rev  = row['reviewerID']
    countU[rev]  += 1
    countI[asin] += 1

print("total")
print("users : " + str(len(countU)) + " items : " + str(len(countI)))

#Map User and item, Create Clean Review

usermap = dict()
usernum = 0
itemmap = dict()
itemnum = 0
dfUserMap = pd.DataFrame(columns=['user_id'])
dfItemMap = pd.DataFrame(columns=['item_id'])
dfReviewClean = pd.DataFrame(columns=['user_id','time','item_id','user_name','review','star_rating','summary'])

usernum = 0
itemnum = 0

print("Begining Mapping and Cleaning")

for idx, row in dfReview.iterrows():
  asin = row['asin']
  rev  = row['reviewerID']
  time = row['unixReviewTime']
  
  if countU[rev] < 5 or countI[asin] < 5: #discard this interaction
    continue

  if rev in dfUserMap.values: #already exist in the Map
    userid = dfUserMap.index[dfUserMap['user_id']==rev].tolist()[0]
    
  else: #non existing User in Map make one
    usernum += 1
    userid = usernum
    newUser = pd.DataFrame({'user_id':[rev]},index=[userid])
    dfUserMap = pd.concat([dfUserMap,newUser])
        
  if asin in dfItemMap.values: #already exist in the Map
    
    itemid = dfItemMap.index[dfItemMap['item_id']==asin].tolist()[0]

  else: #non existing Item in Map make one
    itemnum += 1
    itemid = itemnum
    newItem = pd.DataFrame({'item_id':[asin]},index=[itemid])
    dfItemMap = pd.concat([dfItemMap,newItem])

  newReviewClean = pd.DataFrame.from_dict({'user_id':[userid],'time':[time],'item_id':[itemid],'user_name':[name],'review':[review],'star_rating':[star_rating],'summary':[summary]})
  dfReviewClean = pd.concat([dfReviewClean,newReviewClean], ignore_index=True)

dfReviewClean.sort_values(by=['user_id','time'], inplace=True)
"""
Reorder review clean for sequential representation, add user's fake or non fake discrimination :
  - from fake review result determine if the user is false or not-false
  - order user, item interaction according to user and time stamp
  - Map User and Item into more reasonable ID (4 to 6 figures)
  - Clean Meta Data and match Item ID with Mapped ID
"""


## results
## user | sequence | fakeLabel

