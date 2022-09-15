import torch
import pandas as pd
import numpy as np
import sys
import copy
import math
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
#import matplotlib.pyploat as plt

path ='beauty_discriminated.csv'
#path ='ReviewCleanToys_and_Games.csv'
#path ='ReviewCleanSports_and_Outdoors.csv'
k = 4 #number of clusters
dim = 5

"""
variables(feature name) :
sequence_length,
number_of_top10_items
number_of_repetition
average_rating  ---- x
average_interval_length ---- x
"""

        
def getRepetition(item_list):
    counter = {}
    for i in set(item_list):
        counter[i] = item_list.count(i)
    repetition = 0  
    for key in counter:
        repetition += ( counter[key] - 1 )
    return repetition

def getNumberOfTop10Items(top10_dict, item_list):
    numberOfTop10Items = 0
    for item in item_list :
        if item in top10_dict :
             numberOfTop10Items += 1        
    return numberOfTop10Items

def getAverageInterval(time_list, last_index):
    interval = 0
    
    for i in range(len(time_list)+last_index-1):
        #print(time_list[i+1], time_list[i])
        interval += time_list[i+1] - time_list[i]
    
    if (len(time_list)+last_index) != 0 :
        return interval/(len(time_list)+last_index)
    else :
        return 0
    
    
def getAverageRating(rating_list, last_index):
    return np.mean(np.array(rating_list[:last_index]))
    

    
### for making train/test
def df_feature_collection(df, is_test):
    """
    from dataset
    collect sequence's feature information for clustering via GMM/K-Means
    NOTE! only sequence that will appeal in training sets
    
    parameters :
    df (DataFrame) : dataset
    is_test (boolean) : generate training set for testing if true, if false for validation
    
    returns :
    user(sequence)'s feature table (DataFrame)
    """
    
    last_item_index = -1
    if is_test:
        last_item_index = -1
    else :
        last_item_index = -2
    
    User         = defaultdict(list)
    User_time    = defaultdict(list)
    User_ratings = defaultdict(list)
    
    item_cnt = defaultdict(lambda: 0)
    user_label = {}
    user_seqLength = {}
    user_train = {}
    user_test  = {}
    
    
    
    user_features = {'sequence_length':[],'number_of_top10_items':[],'number_of_repetition':[],'average_rating':[],'average_interval_length':[]}
    
    usernum = 0
    itemnum = 0
    
    for idx, row in df.iterrows():
        u = row['user_id']
        i = row['item_id']
        t = row['time']
        rating = row['star_rating']
        item_cnt[i] += 1
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
        User_time[u].append(t)
        User_ratings[u].append(rating)
        user_label[u] = [row['user_fake_label_rule1'],row['user_fake_label_rule2']]
    
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_test[user] = []
        else:
            user_train[user] = User[user][:last_item_index]
            user_test[user] = []
            item_cnt[User[user][last_item_index]] += -1
            user_test[user].append(User[user][last_item_index])
    
    top10Items = dict(sorted(item_cnt.iteritems(), key=operator.itemgetter(1), reverse=True)[:10])
    
    for user in user_train:
        user_features['sequence_length'].append(len(user_train[user]))
        user_features['number_of_top10_items'].append(getNumberOfTop10Items(top10Items,user_train[user]))
        user_features['number_of_repetition'].append(getRepetition(user_train[user]))
        user_features['average_rating'].append(getAverageRating(User_ratings[user],last_item_index))
        user_features['average_interval_length'].append(getAverageInterval(User_time[user],last_item_index))   

    return pd.DataFrame(user_features)

### for making train/test
def array_feature_collection(df, is_test):
    """
    from dataset
    collect sequence's feature information for clustering via GMM/K-Means
    NOTE! only sequence that will appeal in training sets
    
    parameters :
    df (DataFrame) : dataset
    is_test (boolean) : generate training set for testing if true, if false for validation
    
    returns :
    user(sequence)'s numpy array
    """
    
    last_item_index = -1
    if is_test:
        last_item_index = -1
    else :
        last_item_index = -2
    
    User         = defaultdict(list)
    User_time    = defaultdict(list)
    User_ratings = defaultdict(list)
    
    user_label = {}
    user_seqLength = {}
    user_train = {}
    user_test  = {}
    
    item_cnt = defaultdict(lambda: 0)
    
    #user_features = {'sequence_length':[],'number_of_top10_items':[],'number_of_repetition':[],'average_rating':[],'average_interval_length':[]}
    user_features = {'sequence_length':[],'number_of_top10_items':[],'number_of_repetition':[]}
    usernum = 0
    itemnum = 0
    
    for idx, row in df.iterrows():
        u = row['user_id']
        i = row['item_id']
        t = row['time']
        rating = row['star_rating']
        item_cnt[i] += 1
        #addItemCnt(item_cnt, i, 1)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
        User_time[u].append(t)
        User_ratings[u].append(rating)
        user_label[u] = [row['user_fake_label_rule1'],row['user_fake_label_rule2']]
    
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_test[user] = []
        else:
            user_train[user] = User[user][:last_item_index]
            user_test[user] = []
            item_cnt[User[user][last_item_index]] += -1
            user_test[user].append(User[user][last_item_index])
    
    top10Items = dict(sorted(item_cnt.items(), key=lambda k_v:k_v[1], reverse=True)[:10])
    #print(top10Items)
    feature_list = []
    feature_list_real_r1 = []
    feature_list_fake_r1 = []
    feature_list_real_r2 = []
    feature_list_fake_r2 = []
    
    
    for user in user_train:
        #new_row = [len(user_train[user]),getNumberOfTop10Items(top10Items,user_train[user]),getRepetition(user_train[user]),getRepetition(user_train[user]),getAverageInterval(User_time[user],last_item_index)]
        new_row = [len(user_train[user]),getNumberOfTop10Items(top10Items,user_train[user]),getRepetition(user_train[user])]
        feature_list.append(new_row)
        
        if user_label[user][0] == 'fake':
            feature_list_fake_r1.append(new_row)
        else :
            feature_list_real_r1.append(new_row)
            
        if user_label[user][1] == 'fake':
            feature_list_fake_r2.append(new_row)
        else :
            feature_list_real_r2.append(new_row)
    
    
    return np.array(feature_list), np.array(feature_list_real_r1), np.array(feature_list_fake_r1), np.array(feature_list_real_r2),np.array(feature_list_fake_r1), user_train, user_test


def array_feature_collection_no_label(df, is_test):
    """
    from dataset
    collect sequence's feature information for clustering via GMM/K-Means
    NOTE! only sequence that will appeal in training sets
    
    parameters :
    df (DataFrame) : dataset
    is_test (boolean) : generate training set for testing if true, if false for validation
    
    returns :
    user(sequence)'s numpy array
    """
    
    last_item_index = -1
    if is_test:
        last_item_index = -1
    else :
        last_item_index = -2
    
    User         = defaultdict(list)
    User_time    = defaultdict(list)
    User_ratings = defaultdict(list)
    
    user_seqLength = {}
    user_train = {}
    user_test  = {}
    
    item_cnt = defaultdict(lambda: 0)
    
    #user_features = {'sequence_length':[],'number_of_top10_items':[],'number_of_repetition':[],'average_rating':[],'average_interval_length':[]}
    user_features = {'sequence_length':[],'number_of_top10_items':[],'number_of_repetition':[]}
    usernum = 0
    itemnum = 0
    
    for idx, row in df.iterrows():
        u = row['user_id']
        i = row['item_id']
        t = row['time']
        rating = row['star_rating']
        item_cnt[i] += 1
        #addItemCnt(item_cnt, i, 1)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
        User_time[u].append(t)
        User_ratings[u].append(rating)

    
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_test[user] = []
        else:
            user_train[user] = User[user][:last_item_index]
            user_test[user] = []
            item_cnt[User[user][last_item_index]] += -1
            user_test[user].append(User[user][last_item_index])
    
    top10Items = dict(sorted(item_cnt.items(), key=lambda k_v:k_v[1], reverse=True)[:10])
    #print(top10Items)
    feature_list = []

    
    
    for user in user_train:
        new_row = [len(user_train[user]),getNumberOfTop10Items(top10Items,user_train[user]),getRepetition(user_train[user]),getRepetition(user_train[user]),getAverageInterval(User_time[user],last_item_index)]
        #new_row = [len(user_train[user]),getNumberOfTop10Items(top10Items,user_train[user]),getRepetition(user_train[user])]
        feature_list.append(new_row)

    
    
    return np.array(feature_list), user_train, user_test


def cluster_kMeans(data,cluster_number):
    km = KMeans(n_clusters=cluster_number)
    cluster_result = km.fit_predict(data)
    
    label = np.array(range(0,k))
    
    label_sqDistances = defaultdict(list)
    label_avg_sqDistances = np.zeros(cluster_number)
    
    for user_id, cluster_label in enumerate(cluster_result):
        label_sqDistances[cluster_label].append(np.square(km.cluster_centers_[cluster_label] - data[user_id]))
    
    for label_index, sqDistances in label_sqDistances.items():
        label_avg_sqDistances[label_index] = np.mean(np.array(sqDistances),  dtype=np.float64)  
        
    return km.inertia_, cluster_result, label_avg_sqDistances



def ngram(sequence, num):
    res = []
    slen = len(sequence) - num + 1
    if slen <= 0:
        return res
    
    for i in range(slen):
        small_sequence = sequence[i:i+num]
        res.append(small_sequence)
    return res

def ngram_similarity(representative_ngram, comparee_ngram):
    cnt = 0
    for i in representative_ngram:
        for j in comparee_ngram:
            if i == j:
                #print('Hit',i,j)
                cnt += 1
    if len(representative_ngram) > 0 :
        return cnt/len(representative_ngram)
    else :
        return 0



def cd_hit(sequences, word_size, threshold):
    max_cluster_cnt = len(sequences)
    clustered_key = defaultdict(list)
    clustered_similarity_avg = defaultdict(lambda: 0.00)
    
    #sort sequence by length
    sorted_sequences = sorted(sequences.items(), key=lambda item:len(item[1]), reverse=True)
    sorted_sequences_ngram = [[item[0],ngram(item[1], word_size)] for item in sorted_sequences]
    
    clustering_sequences_ngram = sorted_sequences_ngram
    for i in range(0,max_cluster_cnt):
        unclustered_ngram = []
        #longest one becomes the representative of the first cluster
        repersentative_ngram = clustering_sequences_ngram[0]
        clustered_key[i].append(repersentative_ngram[0])
        #compare remaining sequence with repersentative, check for similarity and group
        for sequence_ngram in clustering_sequences_ngram[1:]:
            similarity = ngram_similarity(repersentative_ngram[1], sequence_ngram[1])
            if  similarity > threshold :
                clustered_key[i].append(sequence_ngram[0])
                clustered_similarity_avg[i] += similarity
            else :
                unclustered_ngram.append(sequence_ngram)
            
        if len(clustered_key[i]) > 1:
            clustered_similarity_avg[i] = clustered_similarity_avg[i]/(len(clustered_key[i])-1)    
        else :
            clustered_similarity_avg[i] = 0
            
        if len(unclustered_ngram) == 0:
            break
            
        clustering_sequences_ngram = unclustered_ngram
        
    return clustered_key, clustered_similarity_avg
 

word_size_parameter = 4
threshold_parameter = 0.7

dfdataset = pd.read_csv(path)
data, data_real_r1, data_fake_r1,data_real_r2,data_fake_r2, train_sequences, test_sets = array_feature_collection(dfdataset,True)

data_feauters_dict = {'all':data,'r1_real':data_real_r1,'r1_fake':data_fake_r1,'r2_real':data_real_r2,'r2_fake':data_fake_r2}

"""
data, train_sequences, test_sets = array_feature_collection_no_label(dfdataset,True)
clusterd_fin, clusterd_avg_similarities = cd_hit(train_sequences, word_size_parameter , threshold_parameter)
top10cluster = dict(sorted(clusterd_fin.items(), key=lambda item:len(item[1]), reverse=True)[:10])
for key in top10cluster:
    print(f'label:{key}|counting :{len(top10cluster[key])}|similarity_avg:{clusterd_avg_similarities[key]:.4F}')
    
    for user_id in top10cluster[key]:
        print(f'user_{user_id}:{train_sequences[user_id]}')



pd.DataFrame.from_dict(top10cluster, orient='index').to_csv('cluster_results.csv')
"""








"""
for i in range(0, len(clusterd_fin)):
    print(f'label:{i}|counting :{len(clusterd_fin[i])}|similarity_avg:{clusterd_avg_similarities[i]:.8F}')
"""
#print(data.size, data_real_r1.size, data_fake_r1.size, data_real_r2.size, data_fake_r2.size)

#gmm = GaussianMixture(n_components=k, random_state=0).fit(data)


for key, data in data_feauters_dict.items():
    print(key)
    inertia, cluster_result, label_inertia = cluster_kMeans(data, 6)
    print(f"k:{6}|inertia={inertia:.4E}|log(inertia)={math.log(inertia):.4F}")
    for inertia_per_label in label_inertia:
        print(inertia_per_label)





def np_ngram(sequence, num):
    res = []
    slen = len(sequence) - num + 1
    if slen <= 0:
        return res
    
    for i in range(slen):
        small_sequence = sequence[i:i+num]
        res.append(small_sequence)
    return np.array(res)

def np_ngram_similarity(representative_ngram, comparee_ngram):
    cnt = 0
    for i in representative_ngram:
        for j in comparee_ngram:
            if i == j:
                cnt += 1
    if len(representative_ngram) > 0 :
        return cnt/len(representative_ngram)
    else :
        return 0



def np_cd_hit(sequences, word_size, threshold):
    max_cluster_cnt = len(sequences)
    clustered_key = defaultdict(list)
    clustered_similarity_avg = defaultdict(lambda: 0.00)
    
    #sort sequence by length
    sorted_sequences = sorted(sequences.items(), key=lambda item:len(item[1]), reverse=True)
    sorted_sequences_ngram = [[item[0],ngram(item[1], word_size)] for item in sorted_sequences]
    
    clustering_sequences_ngram = sorted_sequences_ngram
    for i in range(0,max_cluster_cnt):
        unclustered_ngram = []
        #longest one becomes the representative of the first cluster
        repersentative_ngram = clustering_sequences_ngram[0]
        clustered_key[i].append(repersentative_ngram[0])
        #compare remaining sequence with repersentative, check for similarity and group
        for sequence_ngram in clustering_sequences_ngram[1:]:
            similarity = ngram_similarity(repersentative_ngram[1], sequence_ngram[1])
            if  similarity > threshold :
                clustered_key[i].append(sequence_ngram[0])
                clustered_similarity_avg[i] += similarity
            else :
                unclustered_ngram.append(sequence_ngram)
            
        if len(clustered_key[i]) > 1:
            clustered_similarity_avg[i] = clustered_similarity_avg[i]/(len(clustered_key[i])-1)    
        else :
            clustered_similarity_avg[i] = 0
            
        if len(unclustered_ngram) == 0:
            break
            
        clustering_sequences_ngram = unclustered_ngram
        
    return clustered_key, clustered_similarity_avg 
