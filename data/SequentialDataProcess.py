import os
import sys
import random
import gzip
import json
import pandas as pd
import numpy as np
import json
from collections import defaultdict
 
# for eval()
true = True
false = False


datasetName = r'reviews_Sports_and_Outdoors'

"""
dataset name
"""

dataset_review = r'raw/'+datasetName+r'.json.gz' 

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


def keyHandle(d, key):
    value = None
    try :
        value = d[key] 
    except KeyError:
        value = None
    
    return value

def getDFMinus(path):
    i = 0
    df = {}
    for d in parse(path):
        overall = keyHandle(d, 'overall')
        unixReviewTime = keyHandle(d, 'unixReviewTime')
        asin = keyHandle(d, 'asin')
        reviewerID = keyHandle(d, 'reviewerID')
        reviewText = keyHandle(d, 'reviewText')
        
        df[i] = {'overall':overall,'asin':asin, 'reviewerID':reviewerID, 'unixReviewTime':unixReviewTime, 'reviewText':reviewText}
        
        i += 1
    return pd.DataFrame.from_dict(df, orient = 'index')

dfReview = getDFMinus(dataset_review) ##this is where all the interactions are at

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
reviewClean = {'user_id':[],'time':[],'item_id':[],'review':[],'star_rating':[]}

print("Begining Mapping and Cleaning")

for idx, row in dfReview.iterrows():
  asin = row['asin']
  rev  = row['reviewerID']
  time = row['unixReviewTime']
  review=row['reviewText']
  star_rating = row['overall']
  
  if countU[rev] < 5 or countI[asin] < 5: #discard this interaction
    continue

  if rev in usermap: #already exist in the Map
    userid = usermap[rev]
    
  else: #non existing User in Map make one
    usernum += 1
    userid = usernum
    usermap[rev] = userid
        
  if asin in itemmap: #already exist in the Map
    itemid = itemmap[asin] 
    
  else: #non existing Item in Map make one
    itemnum += 1
    itemid = itemnum
    itemmap[asin] = itemid
  
  reviewClean['user_id'].append(userid)
  reviewClean['item_id'].append(itemid)
  reviewClean['time'].append(time)
  reviewClean['review'].append(review)
  reviewClean['star_rating'].append(star_rating)

print("All Cleaned! Exporting")
dfReviewClean = pd.DataFrame(reviewClean)
dfReviewClean.sort_values(by=['user_id','time'], inplace=True)
dfReviewClean.to_csv("ReviewClean"+datasetName+".csv")



