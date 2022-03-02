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
#python spcotmhp_viterbi_metric.py trialname mapname iteration sample type_gauss
#python spcotmhp_viterbi_metric.py 3LDK_01 s3LDK_01 1 0 n

import os
import sys
import time
#from math import log
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

#重みはガウスか正規化ガウスか
type_gauss  = sys.argv[5] # g: gauss, ng: normalized gauss

#Request the index number of the robot initial position
#init_position_num = sys.argv[5] #0

#Request the file number of the speech instruction   
#speech_num = sys.argv[6] #0



# For saveing the metric path
Like_save     = [ [0.0 for atem in range(K)] for aky in range(K) ]
Distance_save = [ [0.0 for atem in range(K)] for aky in range(K) ]


##FullPath of folder
filename = outputfolder_SIG + trialname #+ "/" 
print(filename, iteration, sample)
outputfile = filename + navigation_folder #+ "viterbi_node/"  #outputfolder + trialname + navigation_folder
if (type_gauss == "g"):
    outputsubfolder = outputfile + "viterbi_node_gauss/"
else:
    outputsubfolder = outputfile + "viterbi_node/"
#outputname = outputfile + "SpCoTMHP"+"S"+str(start)+"G"+str(speech_num)

Makedir( outputfile )
Makedir( outputsubfolder )


#Read the files of learned parameters  #THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
THETA = read_data.ReadParameters(iteration, sample, filename, trialname)
W, W_index, Mu, Sig, Pi, Phi_l, K, L = THETA
#W_index = THETA[1]

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


for st_i in range(K):
 for gl_i in range(K):
  if st_i == gl_i:
      Distance_save[st_i][gl_i]=0
      Like_save[st_i][gl_i]=0
  elif CoonectMatricx[st_i][gl_i] == 1:
      St=st_i
      Gl=gl_i

      outputname = outputsubfolder + "SpCoTMHP_"+"S"+str(St)+"_G"+str(Gl)
      #PathWeightMap = PostProbMap_nparray_jit(CostMapProb,Mu,Sig,map_length,map_width,Gl)
      
      # スタート：ガウス平均
      sta=tools.Map_coordinates_To_Array_index(Mu[St])
      #print("s_n"+str(s_n)+"g_n"+str(g_n))
      start=[sta[1],sta[0]] #スタート位置を指定
      #print(str(start))

      if (SAVE_time == 1):
        #Substitution of start time
        start_time = time.time() 

      #########
      #Path-Planning
      Path, Path_ROS, PathWeightMap, Path_one = path_calculate.ViterbiPathPlanner(start, THETA, CostMapProb, outputfile, outputname,Gl,St, type_gauss) #gridmap, costmap)

      if (SAVE_time == 1):
          #PP終了時刻を保持
          end_pp_time = time.time()
          time_pp = end_pp_time - start_time 
          fp = open( outputname + "_time_pp.txt", 'w')
          fp.write(str(time_pp)+"\n")
          fp.close()

      #The moving distance of the path
      Distance = tools.PathDistance(Path_one)
      Distance_save[st_i][gl_i] = Distance

      #Save the moving distance of the path
      save_data.SavePathDistance(Distance, outputname)
      print("Path distance using Viterbi algorithm is "+ str(Distance))

      #Save the path
      save_data.SavePath(start, [], Path, Path_ROS, outputname)


      #Save the PathWeightMap(PathPlanner内部で実行)
      #####save_data.SaveProbMap(PathWeightMap, outputname)


      #Save the log-likelihood of the path
      #PathWeightMapとPathからlog likelihoodの値を再計算する
      LogLikelihood_step = np.zeros(T_horizon)
      LogLikelihood_sum  = np.zeros(T_horizon)

      for t in range(T_horizon):
          #print PathWeightMap.shape, Path[t][0], Path[t][1]
          LogLikelihood_step[t] = np.log(PathWeightMap[ Path[t][0] ][ Path[t][1] ])

          if (t == 0):
              LogLikelihood_sum[t] = LogLikelihood_step[t]
          elif (t >= 1):
              LogLikelihood_sum[t] = LogLikelihood_sum[t-1] + LogLikelihood_step[t]
            
      Like_save[st_i][gl_i] = LogLikelihood_sum[-1]

      #すべてのステップにおけるlog likelihoodの値を保存
      save_data.SaveLogLikelihood(LogLikelihood_step,0,0,outputname)

      #すべてのステップにおける累積報酬（sum log likelihood）の値を保存
      save_data.SaveLogLikelihood(LogLikelihood_sum,1,0,outputname)

  else:
      Distance_save[st_i][gl_i]=0
      Like_save[st_i][gl_i]=0

print("[END] SpCoTMHP. (Viterbi metric path)")

if (type_gauss == "g"):
    outputsubfolder = outputfile + "Viterbi_SpCoTMHP_gauss_"
else:
    outputsubfolder = outputfile + "Viterbi_SpCoTMHP_"

np.savetxt(outputname+"distance.csv",Distance_save,delimiter=",")
np.savetxt(outputname+"likelihood.csv",Like_save,delimiter=",")
