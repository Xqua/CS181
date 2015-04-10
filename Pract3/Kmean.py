#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from scipy import sparse
from collections import Counter


def Build_Matrix(df):
    mat = np.zeros((N_users, N_artists))
    for i in range(len(df)):
        if i % 10000 == 0:
            print len(df) - i
        u = df['user'][i]
        a = df['artist'][i]
        ui = users[u]
        ai = artists[a]
        mat[ui][ai] = df['plays'][i] / float(N_Play_users[u])
    return sparse.csr_matrix(mat)

print "loading Dataset"
df = pd.read_csv('train.csv')
print "Dataset Loaded"
print "Creating dictionaries"
users_l = sorted(df['user'].unique())
artists_l = sorted(df['artist'].unique())
N_users = len(users_l)
N_artists = len(artists_l)
users = {}
artists = {}
for i in range(N_users):
    users[users_l[i]] = i
for i in range(N_artists):
    artists[artists_l[i]] = i

N_Play_users = Counter()
for i in range(len(df)):
    u = df['user'][i]
    N_Play_users[u] += df['plays'][i]
print "Created !"

print "Matrix built"
Mat = Build_Matrix(df)

print "fitting..."
clf = MiniBatchKMeans(n_clusters=100, init='k-means++', max_iter=100, batch_size=100, verbose=0, compute_labels=True, random_state=None, tol=0.0, max_no_improvement=10, init_size=None, n_init=3, reassignment_ratio=0.01)
clf.fit(Mat)
lab = clf.labels_
print "Fitted !"

dic = {}
uid2C = {}

print "Sorting"
for i in range(Mat.shape[0]):
    k = lab[i]
    if k not in dic:
        dic[k] = []
    uid2C[i] = k
    dic[k].append(Mat[i])

for k in dic.keys():
    dic[k] = np.sum(dic[k], axis=0) / len(dic[k])

print "loading test data"
df_test = pd.read_csv('test.csv')

print "Writting result"
f = open('result.csv','w')
f.write('Id,plays\n')

for i in range(len(df_test)):
    u = df_test['user'][i]
    a = df_test['artist'][i]
    ID = df_test['Id'][i]
    ui = users[u]
    ai = artists[a]
    k = uid2C[ui]
    # print k,ai
    pred = dic[k].toarray()[0][ai] * N_Play_users[u]
    f.write('%s,%s\n' % (ID, pred))

f.close()
