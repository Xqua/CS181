#!/usr/bin/python

import CracktheCode
import numpy as np

f = open('results_elasticnet.tsv','w')
R, RMSE = [],[]
for i in range(20):
	print "Bootstrap nb", i
	M = CracktheCode.Models('conso.csv', maxrow=100000)
	w = M.Bayesian()
	r = M.rsquare()
	rmse = M.RMSE()
	R.append(r)
	RMSE.append(rmse)
	print "r:%s, rmse:%s \n mean r:%s, std r:%s \nmean RMSE:%s, std RMSE:%s" % (r, rmse, np.mean(R), np.std(R), np.mean(RMSE), np.std(RMSE))
	f.write('%s\t%s\t%s\n' % (r,rmse,w))
