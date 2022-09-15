import sys
import copy
import torch
import random
import math
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

import pandas as pd


# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


def sample_function_fr(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train['item_ids'][user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        rsq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        prs = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nrs = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train['item_ids'][user][-1]
        nxtr= user_train['review_ids'][user][-1]
        idx_item = maxlen - 1
        idx_review = maxlen - 1
         
        ts = set(user_train['item_ids'][user])
        
        for i in reversed(user_train['item_ids'][user][:-1]):
            seq[idx_item] = i
            pos[idx_item] = nxt
            if nxt != 0: neg[idx_item] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx_item -= 1
            if idx_item == -1: break

        
        for r in reversed(user_train['review_ids'][user][:-1]):
            rsq[idx_review] = r
            prs[idx_review] = nxtr
            if nxtr != 0: nrs[idx_review] = np.random.randint(1, 2, 1)
            nxtr = r
            idx_review -= 1
            if idx_review == -1: break

        return (user, seq, rsq, pos, prs, neg ,nrs)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))

class WarpSampler_fr(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function_fr, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

def df_data_partition(df, is_valid = False):
    usernum = 0
    itemnum = 0
    
    fake_review_cnt = 0
    real_review_cnt = 0
    
    User = defaultdict(list)
    User_review = defaultdict(list)
    final_idx = -1
    if is_valid:
        final_idx = -2
    
    user_train = {'item_ids':{},'review_ids':{}}
    user_test  = {'item_ids':{},'review_ids':{}}
    
    for idx, row in df.iterrows():
        u = row['user_id']
        i = row['item_id']
        
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
        
        if row['fake_review'] == 'fake':
            User_review[u].append(1)
            fake_review_cnt += 1
        else :
            User_review[u].append(2)
            real_review_cnt += 1
        
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 2:
            user_train['item_ids'][user] = User[user]
            user_train['review_ids'][user] = User_review[user]
            user_test['item_ids'][user] = []
            user_test['review_ids'][user] = []
        else:
            user_train['item_ids'][user] = User[user][:final_idx]
            user_train['review_ids'][user] = User_review[user][:final_idx]
            user_test['item_ids'][user] = []
            user_test['item_ids'][user].append(User[user][final_idx])
            user_test['review_ids'][user] = []
            user_test['review_ids'][user].append(User_review[user][final_idx])    
    
    print(f'number of fake review:{fake_review_cnt}, number of real review:{real_review_cnt}')
    return [user_train, user_test, usernum, itemnum]



   
# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

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

def randomly_drop(cluster_key_list, scaling_rate):
    num_key = len(cluster_key_list)
    end_of_the_line = math.floor(num_key * scaling_rate)
    randomized = cluster_key_list
    random.shuffle(randomized)
    return randomized[:end_of_the_line]
  
def cluster_and_scale(train_set, word_size, threshold, minimun_cluster_size, scaling_rate):
    """
    from train_set cluster and scale via scaling rate
    randomly discard sequences via scaling rate
    parameters :
        train_set (dict) : training sequences 
        word_size (int) : word size(n-gram) for similarity check
        threshold (float) : similarity threshold for clustering
        minimum_cluster_size (boolean) : cluster that containing more sequence than minimum_cluster_size gets scaled 
        scaling_rate (float) : drop (1-scaling_rate) of sequence in cluster randomly
    returns :
        scaled_train (dict) : scaled training set 
    """
    clusterd_key, clusterd_avg_similarities = cd_hit(train_set, word_size , threshold)
    clusterd_key_sorted = dict(sorted(clusterd_key.items(), key=lambda item:len(item[1]), reverse=True))

    scaled_train = defaultdict(list)
    
    for cluster_label in clusterd_key_sorted:
        if len(clusterd_key_sorted[cluster_label]) < minimun_cluster_size:
            for user_id in clusterd_key_sorted[cluster_label]:
                  scaled_train[user_id] = train_set[user_id]
        else :
            using_user_id_list = randomly_drop(clusterd_key_sorted[cluster_label], scaling_rate)
            for user_id in using_user_id_list:
                scaled_train[user_id] = train_set[user_id]
    
    return scaled_train, clusterd_key, clusterd_avg_similarities


def evaluate_via_user(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user 
    
    
# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate_new(model, dataset, maxlen):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train['item_ids'][u]) < 1 or len(test['item_ids'][u]) < 1: continue

        seq = np.zeros([maxlen], dtype=np.int32)
        rsq = np.zeros([maxlen], dtype=np.int32)
        idx_item = maxlen - 1
        idx_review = maxlen - 1
        
        seq[idx_item] = valid['item_ids'][u][0]
        rsq[idx_review] = valid['review_ids'][u][0]
        idx_item -= 1
        idx_review -=1
        for i in reversed(train['item_ids'][u]):
            seq[idx] = i
            idx_item -= 1
            if idx_item == -1: break
        for r in reversed(train['review_ids'][user][:-1]):
            rsq[idx_review] = r
            idx_review -= 1
            if idx_review == -1: break    
        
        rated = set(train['item_ids'][u])
        rated.add(0)
        item_idx = [test['item_ids'][u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid_new(model, dataset, maxlen):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train['item_ids'][u]) < 1 or len(valid['item_ids'][u]) < 1: continue

        seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(train['item_ids'][u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train['item_ids'][u])
        rated.add(0)
        item_idx = [valid['item_ids'][u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
    
    
def evaluation(model, dataset, maxlen, device):
    [train, test, usernum, itemnum] =  copy.deepcopy(dataset)
    
    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    
    if usernum > 10000 :
        users = random.sample(range(1, usernum + 1), 10000)
        
    else :
        users = range(1,usernum + 1)
    
    
    for u in users:
        if len(train['item_ids'][u]) < 1 or len(test['item_ids'][u]) < 1: continue
        
        seq = np.zeros([maxlen], dtype=np.int32)
        rsq = np.zeros([maxlen], dtype=np.int32)
        idx_item   = maxlen - 1
        idx_review = maxlen - 1
        
        for i in reversed(train['item_ids'][u]):
            seq[idx_item] = i
            idx_item -= 1
            if idx_item == -1:break
        
        for r in reversed(train['review_ids'][u]):
            rsq[idx_review] = r
            idx_review -= 1
            if idx_review == -1: break 
        
        rated = set(train['item_ids'][u])
        rated.add(0)
        candidate_items = [test['item_ids'][u][0]]
        
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            candidate_items.append(t)
        
        u   = torch.LongTensor(u).to(device)
        seq = torch.LongTensor(seq.reshape(1, -1)).to(device)
        rsq = torch.LongTensor(rsq.reshape(1, -1)).to(device)
        
        predictions = -model.predict(u, seq, rsq, torch.LongTensor(candidate_items).to(device))
        
        #print(candidate_items)
        #print(predictions)
        #print(predictions.size())
        #print(predictions.argsort())
        #print(predictions.argsort().argsort())
        
        rank = predictions.argsort().argsort()[0].item()
        #print(rank)
        
        #raise('stop')
        #rank = predictions[0].argsort().argsort()[0].item()

        valid_user += 1
        
        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

def get_binary_label(rsq):
    fake_label = 0
    
    ## 1 == fake, 2 == real
    if np.count_nonzero(rsq == 1) > np.count_nonzero(rsq==2):
        fake_label = 1
    else : 
        fake_label = 2
    
    return fake_label

def get_frequency_label(rsq):
    
    fake_label = np.count_nonzero(rsq == 1)
    
    return fake_label
    
def get_ratio_label(rsq):
    
    fake_label = np.count_nonzero(rsq == 1) / (np.count_nonzero(rsq == 1) + np.count_nonzero(rsq== 2)) * 10
    fake_label = np.floor(fake_label).astype(int)
    
    return fake_label

def evaluation_with_label(model, dataset, maxlen, device):
    [train, test, usernum, itemnum] =  copy.deepcopy(dataset)
    
    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    
    User_TopN = defaultdict(list)
    
    if usernum > 10000 :
        users = random.sample(range(1, usernum + 1), 10000)
        
    else :
        users = range(1,usernum + 1)
    
    Binary_Result = defaultdict(list)
    Binary_Metric = {}
    Frequency_Result = defaultdict(list)
    Frequency_Metric = {}
    Ratio_Result = defaultdict(list)
    Ratio_Metric = {}
    
    userResults = {}
    
    for u in users:
        if len(train['item_ids'][u]) < 1 or len(test['item_ids'][u]) < 1: continue
        
        seq = np.zeros([maxlen], dtype=np.int32)
        rsq = np.zeros([maxlen], dtype=np.int32)
        idx_item   = maxlen - 1
        idx_review = maxlen - 1
        
        for i in reversed(train['item_ids'][u]):
            seq[idx_item] = i
            idx_item -= 1
            if idx_item == -1:break
        
        for r in reversed(train['review_ids'][u]):
            rsq[idx_review] = r
            idx_review -= 1
            if idx_review == -1: break 
        
        rated = set(train['item_ids'][u])
        rated.add(0)
        candidate_items = [test['item_ids'][u][0]]
        ## 1 == fake, 2 == not fake
        user_real_reviews = np.count_nonzero(rsq == 2)
        user_fake_reviews = np.count_nonzero(rsq == 1)
        
        user_label_B = get_binary_label(rsq)
        user_label_F = get_frequency_label(rsq)
        user_label_R = get_ratio_label(rsq)
        
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            candidate_items.append(t)
        user_id = u
        u   = torch.LongTensor(u).to(device)
        seq = torch.LongTensor(seq.reshape(1, -1)).to(device)
        rsq = torch.LongTensor(rsq.reshape(1, -1)).to(device)
        
        predictions = -model.predict(u, seq, rsq, torch.LongTensor(candidate_items).to(device))
        
        
        rank = predictions.argsort().argsort()[0].item()
        valid_user += 1
        
        user_HIT = 0.0
        user_NDCG = 0.0
        
        if rank < 10:
            user_HIT = 1
            user_NDCG = 1 / np.log2(rank + 2)
            NDCG += user_NDCG
            HT += user_HIT
            Binary_Result[user_label_B].append([1,1/np.log2(rank+2)])
            Frequency_Result[user_label_F].append([1,1/np.log2(rank+2)])
            Ratio_Result[user_label_R].append([1,1/np.log2(rank+2)])
            

        else:
            Binary_Result[user_label_B].append([0,0])
            Frequency_Result[user_label_F].append([0,0])
            Ratio_Result[user_label_R].append([0,0])
        
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
        
        #print(u, rank, user_HIT, user_NDCG, user_real_reviews, user_fake_reviews)
        userResults[user_id] = [rank, user_HIT, user_NDCG, user_label_B, user_label_F, user_label_R]
        
        
    for label in  Binary_Result:
        user_num_in_label = len(Binary_Result[label])
        label_HT, label_NDCG = 0, 0
        
        for single_result in Binary_Result[label]:
            label_HT   += single_result[0]
            label_NDCG += single_result[1]
            
        Binary_Metric[label] = [label_HT/user_num_in_label, label_NDCG/user_num_in_label, user_num_in_label]
        
    for label in  Frequency_Result:
        user_num_in_label = len(Frequency_Result[label])
        label_HT, label_NDCG = 0, 0
        
        for single_result in Frequency_Result[label]:
            label_HT   += single_result[0]
            label_NDCG += single_result[1]
            
        Frequency_Metric[label] = [label_HT/user_num_in_label, label_NDCG/user_num_in_label, user_num_in_label]
    
    for label in  Ratio_Result:
        user_num_in_label = len(Ratio_Result[label])
        label_HT, label_NDCG = 0, 0
        
        for single_result in Ratio_Result[label]:
            label_HT   += single_result[0]
            label_NDCG += single_result[1]
            
        Ratio_Metric[label] = [label_HT/user_num_in_label, label_NDCG/user_num_in_label, user_num_in_label]
        
    return NDCG / valid_user, HT / valid_user , userResults, dict(sorted(Binary_Metric.items())), dict(sorted(Frequency_Metric.items())), dict(sorted(Ratio_Metric.items()))



    