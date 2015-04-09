#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import prettyplotlib as ppl
from collections import Counter

# Comment aded by yiqun


class MLearn:

    def __init__(self, filename='train.csv', filename_test='test.csv', N_point=None, test=True):
        self.df = pd.read_csv(filename)
        if N_point:
            self.df = self.df[:N_point]
        self.users_l = sorted(self.df['user'].unique())
        self.N_users = len(self.users_l)
        self.artists_l = sorted(self.df['artist'].unique())
        self.N_artists = len(self.artists_l)
        self.users = {}
        self.artists = {}
        self.N_Play_users = Counter()
        for i in range(len(self.df)):
            u = self.df['user'][i]
            self.N_Play_users[u] += self.df['plays'][i]
        for i in range(self.N_users):
            self.users[self.users_l[i]] = i
        for i in range(self.N_artists):
            self.artists[self.artists_l[i]] = i
        # self.matrix = self.Build_Matrix()
        self.dataset = self.Build_Dictionary()
        if test:
            self.dataset, self.test_dataset = self.Split_dataset()
        else:
            self.df_test = pd.read_csv(filename_test)
            self.test_dataset = self.Build_Dictionary(test=True)
        # self.mat_p, self.mat_e = self.Build_Pearson_Euclidian_Matrix
        self.prediction = {}

    def Build_Prediction(self):
        i = 0
        tot = len(self.dataset.keys())
        for user in self.dataset.keys():
            if i % 1000 == 0:
                print "Done %s users, %s to go ..." % (i, tot - i)
            self.prediction[user] = self.User_cond_proba(user)
            i += 1

    def Score_Prediction(self):
        RMSE = 0.0
        tot = 0.0
        for user in self.test_dataset.keys():
            for artist in self.test_dataset[user].keys():
                if artist in self.prediction[user]:
                    N = self.prediction[user][artist]
                else:
                    N = 0.0
                tot += 1
                RMSE = abs(self.test_dataset[user][artist] - N)
        RMSE = RMSE / tot
        return RMSE

    def Split_dataset(self):
        ds, test = {}, {}
        for u in self.users_l:
            if len(self.dataset[u].keys()) > 1:
                a = np.random.choice(self.dataset[u].keys())
                test[u] = {}
                test[u][a] = self.dataset[u][a]
                for artist in self.dataset[u].keys():
                    if artist != a:
                        if u not in ds:
                            ds[u] = {}
                        ds[u][artist] = self.dataset[u][artist]
            else:
                for artist in self.dataset[u].keys():
                    if u not in ds:
                        ds[u] = {}
                    ds[u][artist] = self.dataset[u][artist]
        return ds, test

    def Build_Matrix(self):
        mat = np.zeros((self.N_users, self.N_artists))
        for i in range(len(self.df)):
            if i % 10000 == 0:
                print len(self.df) - i
            u = self.df['user'][i]
            a = self.df['artist'][i]
            ui = self.users[u]
            ai = self.artists[a]
            mat[ui][ai] = self.df['plays'][i]
        return mat

    def Build_Dictionary(self, test=False):
        dic = {}
        for i in range(len(self.df)):
            if i % 10000 == 0:
                print len(self.df) - i, "to go ..."
            if test:
                u = self.df_test['user'][i]
                a = self.df_test['artist'][i]
            else:
                u = self.df['user'][i]
                a = self.df['artist'][i]
            if u not in dic:
                dic[u] = {}
            if test:
                dic[u][a] = 0
            else:
                dic[u][a] = self.df['plays'][i]
        return dic

    def Plot_Histogram(self):
        c = Counter()
        for i in range(len(self.df)):
            u = self.df['user'][i]
            c[u] += self.df['plays'][i]
        res = []
        for u in c.keys():
            res.append(c[u])
        ppl.hist(np.log10(res), bins=100)
        plt.xlabel('Log10(Number of play)')
        plt.ylabel('N')
        plt.show()

    def User_cond_proba(self, user1, method='Pearson'):
        p_u1 = []
        res = Counter()
        for user2 in self.users_l:
            if method == 'Pearson':
                p = self.sim_pearson(user1, user2)
            elif method == 'Euclidian':
                p = self.sim_distance(user1, user2)
            else:
                raise NameError('Method has to be either Pearson of Euclidian')
            p_u1.append((p, user2))
        p_u1 = sorted(p_u1, key=lambda pu: pu[0], reverse=True)
        tot_p = 0
        for pu in p_u1:
            p = pu[0]
            if p != 0:
                tot_p += p
                user2 = pu[1]
                for artist in self.dataset[user2].keys():
                    if artist not in res:
                        res[artist] = []
                    res[artist].append((self.dataset[user2][artist] / float(self.N_Play_users[user2])) * p)
        tot_p = float(tot_p)
        for artist in res.keys():
            score = round(np.sum(res[artist]) / tot_p * self.N_Play_users[user1], 0)  # Here Expected value, (think of MEDIAN or MEAN that is the Question)
            if score < 1:
                del res[artist]
            else:
                res[artist] = score
        return res

    def Build_Pearson_Euclidian_Matrix(self):
        mat_p = np.zeros((self.N_users, self.N_users))
        mat_e = np.zeros((self.N_users, self.N_users))
        for u1 in self.users_l:
            for u2 in self.users_l:
                p = self.sim_pearson(u1, u2)
                e = self.sim_distance(u1, u2)
                mat_p[self.users[u1]][self.users[u1]] = p
                mat_e[self.users[u1]][self.users[u1]] = e
        return mat_p, mat_e

    def sim_distance(self, user1, user2):
        # Get the list of shared_items
        si = {}
        for item in self.dataset[user1]:
            if item in self.dataset[user2]:
                si[item] = 1
        # if they have no ratings in common, return 0
        if len(si) == 0:
            return 0
        # Add up the squares of all the differences
        sum_of_squares = sum([pow((self.dataset[user1][item] / self.N_Play_users[user1]) - (self.dataset[user2][item] / self.N_Play_users[user2]), 2) for item in self.dataset[user1] if item in self.dataset[user2]])
        return 1 / (1 + sum_of_squares)

    def sim_pearson(self, user1, user2):
        # Get the list of mutually rated items
        si = {}
        for item in self.dataset[user1]:
            if item in self.dataset[user2]:
                si[item] = 1
        # Find the number of elements
        n = float(len(si))
        # if they are no ratings in common, return 0
        if n == 0:
            return 0
        # Add up all the preferences
        sum1 = sum([self.dataset[user1][it] for it in si])
        sum2 = sum([self.dataset[user2][it] for it in si])
        # Sum up the squares
        sum1Sq = sum([pow(self.dataset[user1][it], 2) for it in si])
        sum2Sq = sum([pow(self.dataset[user2][it], 2) for it in si])
        # Sum up the products
        pSum = sum([self.dataset[user1][it] * self.dataset[user2][it] for it in si])
        # Calculate Pearson score
        num = pSum - ((sum1 * sum2) / n)
        den = np.sqrt((sum1Sq - pow(sum1, 2) / n) * (sum2Sq - pow(sum2, 2) / n))
        if den == 0:
            return 0.0
        r = num / den
        return abs(r)

M = MLearn(N_point=2000000)
# M = MLearn()
M.Build_Prediction()
S = M.Score_Prediction()
print S
# c = M.User_cond_proba(M.users_l[3])
