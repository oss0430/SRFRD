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

dataset_review = r'realraw/'+datasetName+r'.json.gz' 
dataset_meta = r'realraw/meta_'+datasetName+r'.json.gz'

print("Loading Review {}".format(datasetName))
"""
load data
"""
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

print("Loading Complete")

"""
data processing :
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
  name = row['reviewerName']
  review=row['reviewText']
  star_rating = row['overall']
  summary= row['summary']
  
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

"""
asinTemp = []
idTemp = []
countTemp = []
## counting reports of each data
for idx, row in dfItemMap.iterrows():
    row_asin = row['item_id']
    countTemp.append(countI[row[row_asin]])    
    asinTemp.append(row_asin)
    idTemp.append(idx)

CountItem = {'item_id':idTemp,'asin':asinTemp,'count':countTemp}
dfCI=pd.DataFrame(CountItem)
dfCI("Count_item.csv")

print('counting complete!')
"""
#Sort clean review according to time
dfReviewClean.sort_values(by=['user_id','time'], inplace=True)

print("clean")
print("users : " + str(usernum) + " items : " + str(itemnum))
dfReviewClean.to_csv("ReviewClean"+datasetName+".csv")
