#!/usr/bin/python 

import CracktheCode

M = CracktheCode.Models('train.csv', maxrow=1000000)

w = M.ElasticNet(alpha=0.0002)

RMSE = M.RMSE()

r = M.rsquare()

f = open('results_enet.tsv','a')
f.write("%s\t%s\t%s\n"%(RMSE,r,w))
f.close()