#!/usr/bin/env python
#coding:utf-8

###########################################################
# SpCoTMHPi: Spatial Concept-based Path-Planning Program for SIGVerse
# Path-Planning Program by Viterbi algorithm 
# Path Selection: maximum log-likelihood in a path trajectory
# Akira Taniguchi 2022/02/07
# Spacial Thanks: Shuya Ito
###########################################################

##Command: 
#python spcotmhp_viterbi_metric.py trialname mapname iteration sample init_position_num speech_num initial_position_x initial_position_y
#python spcotmhp_viterbi_metric.py 3LDK_01 s1DK_01 1 0 0 7 100 100 

import os
import sys
import time
from math import log
#import matplotlib.pyplot as plt
import spconavi_read_data
import spconavi_save_data
import spcotmhp_viterbi_path_calculate
from __init__ import *
from submodules import *

tools     = spconavi_read_data.Tools()
read_data = spconavi_read_data.ReadingData()
save_data = spconavi_save_data.SavingData()
path_calculate = spcotmhp_viterbi_path_calculate.PathPlanner()


#if __name__ == '__main__': 
#################################################
print("[START] SpCoTMHP. (Viterbi metric path)")

#map dataの入った部屋環境folder name（学習済みparameter folder name） is requested
#Request a folder name for learned parameters.
trialname = sys.argv[1]

#map file name is requested
mapname = sys.argv[2]

#Request iteration value
iteration = sys.argv[3] #1

#Request sample value
sample = sys.argv[4] #0

#Request the index number of the robot initial position
init_position_num = sys.argv[5] #0

#Request the file number of the speech instruction   
speech_num = sys.argv[6] #0


if (SAVE_time == 1):
  #Substitution of start time
  start_time = time.time()

Like_save=[ [0.0 for atem in range(K)] for aky in range(K) ]
Distance_save=[ [0.0 for atem in range(K)] for aky in range(K) ]

##FullPath of folder
filename = outputfolder_SIG + trialname #+ "/" 
print(filename, iteration, sample)
outputfile = filename + navigation_folder + "viterbi_node/"  #outputfolder + trialname + navigation_folder
#outputname = outputfile + "SpCoTMHP"+"S"+str(start)+"G"+str(speech_num)

#Makedir( outputfolder + trialname )
Makedir( outputfile )
#Makedir( outputname )

#Read the files of learned parameters  #THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
THETA = read_data.ReadParameters(iteration, sample, filename, trialname)
W, W_index, Mu, Sig, Pi, Phi_l, K, L = THETA
#W_index = THETA[1]

if (os.path.isfile(outputfile + "CostMapProb.csv") == False):  #すでにファイルがあれば計算しない
    print("If you do not have map.csv, please run commands for cost map acquisition procedure in advance.")
    ##Read the map file
    gridmap = read_data.ReadMap(outputfile)
    ##Read the cost map file
    costmap = read_data.ReadCostMap(outputfile)

    #Change the costmap to the probabilistic costmap
    CostMapProb = path_calculate.CostMapProb_jit(gridmap, costmap)
    #Write the probabilistic cost map file
    save_data.SaveCostMapProb(CostMapProb, outputfile)
else:
    #Read the probabilistic cost map file
    CostMapProb = read_data.ReadCostMapProb(outputfile)

##Read the speech file
#speech_file = ReadSpeech(int(speech_num))
BoW = [Goal_Word[int(speech_num)]]
#if ( "AND" in BoW ):
#  BoW = Example_AND
#elif ( "OR" in BoW ):
#  BoW = Example_OR

Otb_B = [int(W_index[i] in BoW) * N_best for i in xrange(len(W_index))]
print("BoW:",  Otb_B)

while (sum(Otb_B) == 0):
  print("[ERROR] BoW is all zero.", W_index)
  word_temp = raw_input("Please word?>")
  Otb_B = [int(W_index[i] == word_temp) * N_best for i in xrange(len(W_index))]
  print("BoW (NEW):",  Otb_B)

#Ito# 遷移確率の低いエッジは計算しないようにするために擬似的にpsi_setting.csvを読み込む
#Ito# psiそのものの確率値ではないことに注意
psi     = [ [0.0 for atem in range(K)] for aky in range(K) ]
c=0
for line in open(filename + "/" + trialname + '_psi_'  + 'setting.csv', 'r'):
    itemList = line[:-1].split(',')
    for i in range(len(itemList)):
        if itemList[i] != "":
          psi[c][i] = float(itemList[i])
    c = c + 1 
    
s_n=0 # start node index
g_n=0 # goal node index

""" #Ito
for stp in range(10):
  #for g_n in range(K):
  if psi[s_n][g_n]==1:
      
      if s_n != g_n : 
      #else : 
"""     
outputname = outputfile + "SpCoTMHP"+"_S"+str(s_n)+"G"+str(g_n)
sta=tools.Map_coordinates_To_Array_index(Mu[s_n])
print("s_n"+str(s_n)+"g_n"+str(g_n))
start=[sta[1],sta[0]] #スタート位置を指定
print(str(start))

  
#########
#Path-Planning
Path, Path_ROS, PathWeightMap, Path_one,Step = path_calculate.PathPlanner(Otb_B, start, THETA, CostMapProb, outputfile, speech_num, outputname,g_n,s_n) #gridmap, costmap)

if (SAVE_time == 1):
    #PP終了時刻を保持
    end_pp_time = time.time()
    time_pp = end_pp_time - start_time #end_recog_time
    fp = open( outputname + "_time_pp.txt", 'w')
    fp.write(str(time_pp)+"\n")
    fp.close()

#The moving distance of the path
Distance = path_calculate.PathDistance(Path_one)

#Save the moving distance of the path
save_data.SavePathDistance(Distance, outputname)
print("Path distance using Viterbi algorithm is "+ str(Distance))

#Save the path
save_data.SavePath(Start_Position[int(init_position_num)], [], Path, Path_ROS, outputname)

#Save the PathWeightMap(PathPlanner内部で実行)
#####save_data.SaveProbMap(PathWeightMap, outputname)

#PathWeightMapとPathからlog likelihoodの値を再計算する
LogLikelihood_step = np.zeros(T_horizon)
LogLikelihood_sum  = np.zeros(T_horizon)

for t in range(T_horizon):
    #print PathWeightMap.shape, Path[t][0], Path[t][1]
    if (t < Step):
      LogLikelihood_step[t] = np.log(PathWeightMap[ Path[t][0] ][ Path[t][1] ])
    else: 
      LogLikelihood_step[t] = np.log(PathWeightMap[ Path[Step-1][0] ][ Path[Step-1][1] ])
    if (t == 0):
        LogLikelihood_sum[t] = LogLikelihood_step[t]
    elif (t >= 1):
        LogLikelihood_sum[t] = LogLikelihood_sum[t-1] + LogLikelihood_step[t]
      
    
#すべてのステップにおけるlog likelihoodの値を保存
save_data.SaveLogLikelihood(LogLikelihood_step,0,0,outputname)

#すべてのステップにおける累積報酬（sum log likelihood）の値を保存
save_data.SaveLogLikelihood(LogLikelihood_sum,1,0,outputname)
  

print("[END] SpCoTMHP. (Viterbi metric path)")


outputname=outputfile+"Viterbi_SpCoTMHP_"
np.savetxt(outputname+"distance.csv",Distance_save,delimiter=",")
np.savetxt(outputname+"cost.csv",Like_save,delimiter=",")
