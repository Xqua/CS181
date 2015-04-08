#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import prettyplotlib as ppl
from collection import Counter

#Comment aded by yiqun

class MLearn:

    def __init__(self, filename='train.csv'):
        self.df = pd.read_csv(filename)
        self.matrix = self.Build_Matrix(self.df)

    def Build_Matrix(self, df):
        users = sorted(df['user'].unique())
        N_users = len(users)
        artists = sorted(df['artist'].unique())
        N_artists = len(artists)
        mat = np.zeros((N_users, N_artists))
        for i in range(len(df)):
            u = df['user'][i]
            a = df['artist'][i]
            ui = users.index(u)
            ai = artists.index(a)
            mat[ui][ai] = df['plays'][i]
        return mat

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

M = MLearn()
