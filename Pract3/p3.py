#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import prettyplotlib as ppl
from collections import Counter

#Comment aded by yiqun

class MLearn:

    def __init__(self, filename='train.csv', N_point=None):
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
        # self.mat_p, self.mat_e = self.Build_Pearson_Euclidian_Matrix
        self.prediction = {}

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

    def Build_Dictionary(self):
        dic = {}
        for i in range(len(self.df)):
            if i % 10000 == 0:
                print len(self.df) - i, "to go ..."
            u = self.df['user'][i]
            a = self.df['artist'][i]
            if u not in dic:
                dic[u] = {}
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
        print p_u1[:100]
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
        print tot_p
        print res
        for artist in res.keys():
            score = np.sum(res[artist]) / tot_p * self.N_Play_users[user1]  # Here Expected value, (think of MEDIAN or MEAN that is the Question)
            print artist, score
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
        # print sum1,sum2
        # Sum up the squares
        sum1Sq = sum([pow(self.dataset[user1][it], 2) for it in si])
        sum2Sq = sum([pow(self.dataset[user2][it], 2) for it in si])
        # print sum1Sq, sum2Sq
        # Sum up the products
        pSum = sum([self.dataset[user1][it] * self.dataset[user2][it] for it in si])
        # print pSum, (sum1 * sum2)
        # Calculate Pearson score
        num = pSum - ((sum1 * sum2) / n)
        den = np.sqrt((sum1Sq - pow(sum1, 2) / n) * (sum2Sq - pow(sum2, 2) / n))
        # print n, num, den
        if den == 0:
            return 0.0
        r = num / den
        return abs(r)

M = MLearn(N_point=1000000)
# M = MLearn()
c = M.User_cond_proba(M.users_l[3])
