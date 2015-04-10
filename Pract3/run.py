#!/usr/bin/env python

import p3

# M = MLearn(N_point=100000)
M = p3.MLearn()
M.Build_Prediction()
S = M.Score_Prediction()
print S
# c = M.User_cond_proba(M.users_l[3])
