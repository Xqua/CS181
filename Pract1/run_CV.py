#!/usr/bin/python 

import CracktheCode

M = CracktheCode.Models('train.csv', maxrow=10000)
M.bootstrap_reset()
w_lasso = M.lstsqr()
RMSE_lasso = M.RMSE()
f = open('results.tsv','a')
f.write("%s\t%s\n"%(RMSE_lasso,w_lasso))
f.close()