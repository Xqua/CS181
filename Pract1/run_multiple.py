#!/usr/bin/python 

import subprocess
import sys


NB_cv = 64
P = []
for i in range(NB_cv):
	P.append(subprocess.Popen(['python',sys.argv[1]]))
	print P

