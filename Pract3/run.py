#!/usr/bin/env python

import p3
import numpy as np

# M = p3.MLearn(filename='small_train.csv', N_point=1000000)
M = p3.MLearn(N_point=30000)
# M = p3.MLearn()
# M.dataset = M.Build_Dictionary()
# M.dataset, M.test_dataset = M.Split_dataset()
# M.prediction = {}

M.Build_Prediction()
f = open('saved.npy', 'w')
np.save(f, M.prediction)
# S = M.Score_Prediction()
# print S
# c = M.User_cond_proba(M.users_l[3])
