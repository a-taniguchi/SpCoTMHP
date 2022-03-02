#!/usr/bin/env python
#coding:utf-8

##Command: 
#python3 run_exp2_spcotmhp_dijkstra.py trialname 
#python3 run_exp2_spcotmhp_dijkstra.py 3LDK_01

from codecs import Codec
import sys
import subprocess
from subprocess import PIPE
import itertools
import spconavi_read_data
from __init__ import *

tools     = spconavi_read_data.Tools()
read_data = spconavi_read_data.ReadingData()

#Request a folder name for learned parameters.
trialname = sys.argv[1]

init_position_num = -1

##FullPath of folder
filename = outputfolder_SIG + trialname #+ "/" 

#Read the files of learned parameters  #THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
THETA = read_data.ReadParameters(1, 0, filename, trialname)
W, W_index, Mu, Sig, Pi, Phi_l, K, L = THETA

print("RUN EXP2: SpCoTMHP (Dijkstra)")

#Ito# 遷移確率の低いエッジは計算しないようにするために擬似的にpsi_setting.csvを読み込む
#Ito# psiそのものの確率値ではないことに注意
CoonectMatricx     = [ [0.0 for atem in range(K)] for aky in range(K) ]
c=0
for line in open(filename + "/" + trialname + '_psi_'  + 'setting.csv', 'r'):
    itemList = line[:-1].split(',')
    for i in range(len(itemList)):
        if itemList[i] != "":
          CoonectMatricx[c][i] = float(itemList[i])
    c = c + 1 


for i,j in itertools.product(range(K), range(K)):
    #print(i,j)
    if (int(CoonectMatricx[i][j]) == 0): #直接の接続関係がないとき
        for tyukan in range(K):
            if (tyukan != i) and (tyukan != j): # 中間がスタートとゴールと同じにならないようにする
                # Mu[j]が対応するWordを推測 p(word | Mu[j]) = sum_c p(word | W[c]) p( j | phi_l[c][j])
                EstimatedWord = W_index[ np.argmax( [ np.sum( W[c][word] * Phi_l[c][j] * Pi[c] for c in range(L) ) for word in range(len(W_index)) ] ) ] 

                #if (EstimatedWord == W_index[s]): #Mu[j]が推定した単語をゴールとする
                #for s in range(len(W_index)): #環境によって与えられた単語が違うため
                st = tools.Map_coordinates_To_Array_index(Mu[i])
                #gl = Mu[]

                speech_num = Goal_Word.index(EstimatedWord)
                print(i,j, st, str(EstimatedWord), speech_num)
                
                
                # Mu[tyukan]が対応するWordを推測 p(word | Mu[tyukan]) = sum_c p(word | W[c]) p( tyukan | phi_l[c][tyukan])
                EstimatedWord_tyukan = W_index[ np.argmax( [ np.sum( W[c][word] * Phi_l[c][tyukan] * Pi[c] for c in range(L) ) for word in range(len(W_index)) ] ) ] 
                tyukan_num = Goal_Word.index(EstimatedWord_tyukan)                
                print("tyukan:", str(EstimatedWord_tyukan), tyukan_num)
                
                if (speech_num != tyukan_num): #中間の単語インデックスとゴールの単語インデックスが同じにならないようにする
                
                    #python3 spcotmhp_dijkstra_execute.py 3LDK_01 1 0 -1 7 100 100 -1 
                    process_call = "python3 spcotmhp_dijkstra_execute.py " + trialname + " 1 0 " + str(init_position_num)  + " " + str(speech_num) + " " + str(st[0]) + " " +  str(st[1]) + " " + str(tyukan_num)
                    print(process_call)
                    #subprocess.Popen("pwd")
                    subprocess.run( process_call, shell=True ) #, stdout=PIPE, stderr=PIPE


print("END EXP2: SpCoTMHP (Dijkstra)")
