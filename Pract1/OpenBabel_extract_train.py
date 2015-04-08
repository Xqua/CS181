
import CracktheCode
import numpy as np
import pybel

def calculate(smiles, qvals):
	print "Calculating Features ..."
	mols = [pybel.readstring("smi", smile) for smile in smiles]
	fp2 = [mol.calcfp(fptype='fp2') for mol in mols] #1024
	fp3 = [mol.calcfp(fptype='fp3') for mol in mols] #210
	fp4 = [mol.calcfp(fptype='fp4') for mol in mols] #301
	maccs = [mol.calcfp(fptype='maccs') for mol in mols] #166
	print "Storing Features"
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
	# print pack[-1]
	print "Saving into file..."
	f = open('openbabel_rdkit_train.csv','a')
	for r in pack:
		# print ','.join([str(i) for i in r[3]])
		#print "%s,%s,%s\n" % (r[0], ','.join([str(i) for i in r[1]]),r[2])
		tmp = "%s,%s,%s\n" % (r[0], ','.join([str(i) for i in r[1]]),r[2])
		# print tmp
		f.write(tmp)
	f.close()

print "Loading the Data ! ..."
step = 100000
D_test = CracktheCode.Data('train.csv', 500000)
f = open('openbabel_rdkit_train.csv','w')
f.close()
for m in np.arange(0,D_test.N,step):
	print m, "->", m+step
	smiles = D_test.smiles[m:m+step]
	qvals = D_test.gapvalue[m:m+step]
	# print smiles[0], rdkit[0],qvals[0]
	calculate(smiles, qvals)

