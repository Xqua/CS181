#!/usr/bin/env python

import os
import pandas as pd
from collections import Counter

l = os.listdir('.')
os.mkdir('./renamed')

YOURFUCKINGMATRIX_FILENAME = 'Matrix.csv'

mat = pd.read_csv(YOURFUCKINGMATRIX_FILENAME).values

l = sorted(l)

CL = ['A', 'B', 'C']
c = Counter()


for i in range(len(l)):
    m = mat[i]
    cl = CL[m.index(1)]
    c[cl] += 1
    # tc = str(c[cl])
    # while len(tc) < 6:
        # tc = '0' + tc
    # os.rename(l[i], cl + '_' + tc + '.tif')
    os.rename(l[i], cl + '_' + l)
