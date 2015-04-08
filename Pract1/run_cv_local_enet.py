#!/usr/bin/python

import CracktheCode
import numpy as np

f = open('results_elasticnet_alpha.tsv','w')
R, RMSE = [],[]
for alpha in [0.1,0.05,0.01,0.005,0.001,0.0001,0.00001]:
	print "alpha", alpha
	M = CracktheCode.Models('conso.csv', maxrow=100000)
	w = M.ElasticNet(alpha=alpha)
	r = M.rsquare()
	rmse = M.RMSE()
	R.append(r)
	RMSE.append(rmse)
	print "r:%s, rmse:%s \n mean r:%s, std r:%s \nmean RMSE:%s, std RMSE:%s" % (r, rmse, np.mean(R), np.std(R), np.mean(RMSE), np.std(RMSE))
	f.write('%s\t%s\t%s\n' % (r,rmse,w))
