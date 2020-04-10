#c-od-i-n-:-ut-f---8
#!/usr/bin/env python
#pre(2017/02/25)->2018/02/05

import sys
import string
#from sklearn.metrics.cluster import adjusted_rand_score
#import matplotlib.pyplot as plt
import numpy as np
#import math

data_name = raw_input("data_name?(**NUM) > ")
data_num1 = raw_input("data_start_num?(DATA***) > ")
data_num2 = raw_input("data_end_num?(DATA***) > ")
N = int(data_num2) - int(data_num1) +1
#filename = raw_input("Read_Ct_filename?(.csv) >")
S = int(data_num1)


ARIc_M = [[] for c in xrange(N)]
ARIi_M = [[] for c in xrange(N)]  
NMIc_M = [[] for c in xrange(N)]
NMIi_M = [[] for c in xrange(N)]  
PARs_M = [[] for c in xrange(N)]
PARw_M = [[] for c in xrange(N)]
L_M = [[] for c in xrange(N)]  
K_M = [[] for c in xrange(N)]
ESEG_M = [[] for c in xrange(N)]
MM = [ np.array([[] for m in xrange(10) ]) for n in xrange(N)]
#MM_M = 

fp = open('./data/Evaluation/' + data_name + '_' + data_num1 + '_' + data_num2 + '_Evaluation.csv', 'w')
fp.write('ARIc,ARIi,NMIc,NMIi,PARs,PARw,L,K,ESEG\n')

i = 0
ARIc_MAX = [[0,0]]
ARIi_MAX = [[0,0]]
NMIc_MAX = [[0,0]]
NMIi_MAX = [[0,0]]
PARs_MAX = [[0,0]]
PARw_MAX = [[0,0]]
L_MAX = [[0,0]]
K_MAX = [[0,0]]
ESEG_MAX = [[0,0]]

for s in range(N):
  i = 0
  for line in open('./data/' + data_name  + str(S+s).zfill(3) + '/'+ data_name  + str(S+s).zfill(3) + '_A_sougo_Evaluation.csv', 'r'):
    itemList = line[:-1].split(',')
    if (i != 0) and (itemList[0] != '') and (i <= 10):
      #print i,itemList
      ARIc_M[s] = ARIc_M[s] + [float(itemList[1])]
      ARIi_M[s] = ARIi_M[s] + [float(itemList[2])]
      NMIc_M[s] = NMIc_M[s] + [float(itemList[3])]
      NMIi_M[s] = NMIi_M[s] + [float(itemList[4])]
      PARs_M[s] = PARs_M[s] + [float(itemList[5])]
      PARw_M[s] = PARw_M[s] + [float(itemList[6])]
      L_M[s] = L_M[s] + [float(itemList[7])]
      K_M[s] = K_M[s] + [float(itemList[8])]
      ESEG_M[s] = ESEG_M[s] + [float(itemList[9])]

    i = i + 1
         
  ARIc_M[s] = np.array(ARIc_M[s])
  ARIi_M[s] = np.array(ARIi_M[s])
  NMIc_M[s] = np.array(NMIc_M[s])
  NMIi_M[s] = np.array(NMIi_M[s])
  PARs_M[s] = np.array(PARs_M[s])
  PARw_M[s] = np.array(PARw_M[s])
  L_M[s] = np.array(L_M[s])
  K_M[s] = np.array(K_M[s])
  ESEG_M[s] = np.array(ESEG_M[s])
  if (ARIc_M[s][-1] > ARIc_MAX[0][1]):
          ARIc_MAX = [[s+1,ARIc_M[s][-1]]] + ARIc_MAX
  if (ARIi_M[s][-1] > ARIi_MAX[0][1]):
          ARIi_MAX = [[s+1,ARIi_M[s][-1]]] + ARIi_MAX
  if (NMIc_M[s][-1] > NMIc_MAX[0][1]):
          NMIc_MAX = [[s+1,NMIc_M[s][-1]]] + NMIc_MAX
  if (NMIi_M[s][-1] > NMIi_MAX[0][1]):
          NMIi_MAX = [[s+1,NMIi_M[s][-1]]] + NMIi_MAX
  if (PARs_M[s][-1] > PARs_MAX[0][1]):
          PARs_MAX = [[s+1,PARs_M[s][-1]]] + PARs_MAX
  if (PARw_M[s][-1] > PARw_MAX[0][1]):
          PARw_MAX = [[s+1,PARw_M[s][-1]]] + PARw_MAX
  if (L_M[s][-1] > L_MAX[0][1]):
          L_MAX = [[s+1,L_M[s][-1]]] + L_MAX
  if (K_M[s][-1] > K_MAX[0][1]):
          K_MAX = [[s+1,K_M[s][-1]]] + K_MAX
  if (ESEG_M[s][-1] > ESEG_MAX[0][1]):
          ESEG_MAX = [[s+1,ESEG_M[s][-1]]] + ESEG_MAX

#print "MI_MAX:",MI_MAX
#print "ARI_MAX:",ARI_MAX
#print "PARw_MAX:",PARw_MAX
#print "PARs_MAX:",PARs_MAX
#print MI_M
#MM_M = sum(MM)/N
ARIc_MM = sum(ARIc_M)/N
ARIi_MM = sum(ARIi_M)/N
NMIc_MM = sum(NMIc_M)/N
NMIi_MM = sum(NMIi_M)/N
PARs_MM = sum(PARs_M)/N
PARw_MM = sum(PARw_M)/N
L_MM = sum(L_M)/N
K_MM = sum(K_M)/N
ESEG_MM = sum(ESEG_M)/N
#print MI_MM
#MI,ARI,PARs,PARw,

for iteration in xrange(len(ARIc_MM)):
  fp.write( str(ARIc_MM[iteration])+','+ str(ARIi_MM[iteration])+','+ str(NMIc_MM[iteration])+','+ str(NMIi_MM[iteration])+','+ str(PARs_MM[iteration])+','+str(PARw_MM[iteration])+','+ str(L_MM[iteration])+','+ str(K_MM[iteration])+','+str(ESEG_MM[iteration]) )
  fp.write('\n')
fp.write('\n')


for iteration in xrange(len(ARIc_MM)):
  ARIc_MS = np.array([ARIc_M[s][iteration] for s in xrange(N)])
  ARIi_MS = np.array([ARIi_M[s][iteration] for s in xrange(N)])
  NMIc_MS = np.array([NMIc_M[s][iteration] for s in xrange(N)])
  NMIi_MS = np.array([NMIi_M[s][iteration] for s in xrange(N)])
  PARs_MS = np.array([PARs_M[s][iteration] for s in xrange(N)])
  PARw_MS = np.array([PARw_M[s][iteration] for s in xrange(N)])
  L_MS = np.array([L_M[s][iteration] for s in xrange(N)])
  K_MS = np.array([K_M[s][iteration] for s in xrange(N)])
  ESEG_MS = np.array([ESEG_M[s][iteration] for s in xrange(N)])
  #print iteration,np.std(MI_MS, ddof=1)
  fp.write( str(np.std(ARIc_MS, ddof=1))+','+ str(np.std(ARIi_MS, ddof=1))+','+str(np.std(NMIc_MS, ddof=1))+','+ str(np.std(NMIi_MS, ddof=1))+','+ str(np.std(PARs_MS, ddof=1))+','+str(np.std(PARw_MS, ddof=1))+','+ str(np.std(L_MS, ddof=1))+','+ str(np.std(K_MS, ddof=1))+','+str(np.std(ESEG_MS, ddof=1)) )
  fp.write('\n')

print "close."
  
fp.close()
