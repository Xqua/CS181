#!/usr/bin/python 

import CracktheCode

M = CracktheCode.Models('train.csv', maxrow=1000000)

w = M.Bayesian()

RMSE = M.RMSE()

r = M.rsquare()

f = open('results_bayesian.tsv','a')
f.write("%s\t%s\t%s\n"%(RMSE,r,w))
f.close()