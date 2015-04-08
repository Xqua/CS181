#!/usr/bin/python

import CracktheCode
import pybel
import multiprocessing
import numpy as np
import sys
import time

def calculate(smiles, qvals, q):
	mols = [pybel.readstring("smi", smile) for smile in smiles]
	fp2 = [mol.calcfp(fptype='fp2') for mol in mols] #1024
	fp3 = [mol.calcfp(fptype='fp3') for mol in mols] #210
	fp4 = [mol.calcfp(fptype='fp4') for mol in mols] #301
	maccs = [mol.calcfp(fptype='maccs') for mol in mols] #166
	features = []
	for mol in range(len(mols)):
		feature = np.zeros(1024+210+301+166)
		for i in fp2[mol].bits:
			feature[i] = 1
		for i in fp3[mol].bits:
			feature[i+1024] = 1
		for i in fp4[mol].bits:
			feature[i+1024+210] = 1
		for i in maccs[mol].bits:
			feature[i+1024+301] = 1
		features.append(feature)
	pack = []
	for i in range(len(smiles)):
		pack.append((smiles[i],features[i],qvals[i]))
	q.put(pack) #pack




D = CracktheCode.Data('train.csv', maxrow=1000000)

pool_size = 5

q = multiprocessing.Queue()
P = []
jobs = 0
res = []
for m in np.arange(0,len(D.smiles),1000):
	print "Doing:", m
	p = multiprocessing.Process(target=calculate, args=(D.smiles[m:m+1000], D.gapvalue[m:m+1000], q))
	p.start()
	P.append(p)
	jobs +=1 
	if jobs % 3 == 0:
		time.sleep(8)
		print "Done:", m

for i in range(jobs):
	tmp = q.get()
	if len(tmp) != 3:
		for i in tmp: #unpack
			res.append(i)
	elif len(tmp) == 3:
		res.append(tmp)

f = open('openbabel_train.csv','w')
for r in res:
	#print "%s,%s,%s\n" % (r[0], ','.join([str(i) for i in r[1]]),r[2])
	tmp = "%s,%s,%s\n" % (r[0], ','.join([str(i) for i in r[1]]),r[2])
	f.write(tmp)
f.close()

print "FINISH WITH CTRL + C "

sys.exit(0)