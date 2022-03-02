#!/usr/bin/env python
#coding:utf-8

###########################################################
# SpCoNavi: Spatial Concept-based Path-Planning Program for SIGVerse
# Path-Planning Program by Viterbi algorithm 
# Path Selection: maximum log-likelihood in a path trajectory
# Akira Taniguchi 2022/02/08
# Spacial Thanks: Shoichi Hasegawa
###########################################################

##Command: 
#python spconavi_viterbi_execute.py trialname mapname iteration sample init_position_num speech_num initial_position_x initial_position_y
#python spconavi_viterbi_execute.py 3LDK_01 s3LDK_01 1 0 0 7 100 100 

import os
import sys
import time
from __init__ import *
from submodules import *
import spconavi_read_data
import spconavi_save_data
import spconavi_viterbi_path_calculate_copy as spconavi_viterbi_path_calculate

tools     = spconavi_read_data.Tools()
read_data = spconavi_read_data.ReadingData()
save_data = spconavi_save_data.SavingData()
path_calculate = spconavi_viterbi_path_calculate.PathPlanner()

#Definition of action (functions in spconavi_read_data)
action_functions = [tools.right, tools.left, tools.up, tools.down, tools.stay] #, migiue, hidariue, migisita, hidarisita]
cost_of_actions  = np.log( np.ones(len(action_functions)) / float(len(action_functions)) ) #[    1/5,    1/5,  1/5,    1/5,    1/5]) #, ,    1,        1,        1,          1]

#################################################
if __name__ == '__main__': 
    print("[START] SpCoNavi. (Viterbi)")

    #Request a folder name for learned parameters.
    trialname = sys.argv[1]

    #map file name is requested
    mapname = sys.argv[2]

    #Request iteration value
    iteration = sys.argv[3] #1

    #Request sample value
    sample = sys.argv[4] #0

    #Request the index number of the robot initial position
    #init_position_num = sys.argv[5] #0

    #Request the file number of the speech instruction   
    speech_num = sys.argv[6] #0


    start = [int(sys.argv[7]), int(sys.argv[8])] #Start_Position[int(init_position_num)]
    start_inv = [ start[1], start[0] ]
    #start[0] = int(sys.argv[7]) #0
    #start[1] = int(sys.argv[8]) #0
    #start = [start_list[0], start_list[1]]
    print("Start:", start)


    if (SAVE_time == 1):
      #Substitution of start time
      start_time = time.time()

    ##FullPath of folder
    filename = outputfolder_SIG + trialname #+ "/" 
    print(filename, iteration, sample)
    outputfile = filename + navigation_folder #outputfolder + trialname + navigation_folder
    outputsubfolder = outputfile + "spconavi_viterbi/"
    outputname = outputsubfolder + "T"+str(T_horizon)+"N"+str(N_best)+"A"+str(Approx)+"S"+str(start)+"G"+str(speech_num)+"/"
    #if (T_restart != 0):
    #  outputname_restart = outputfile + "T"+str(T_restart)+"N"+str(N_best)+"A"+str(Approx)+"S"+str(start)+"G"+str(speech_num)+"/"

    Makedir( outputfile )
    Makedir( outputsubfolder )
    Makedir( outputname )

    #Read the files of learned parameters  #THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
    THETA = read_data.ReadParameters(iteration, sample, filename, trialname)
    W_index = THETA[1]


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

    Otb_B = [int(W_index[i] in BoW) * N_best for i in range(len(W_index))]
    print("BoW:",  Otb_B)

    #while (sum(Otb_B) == 0):
    #  print("[ERROR] BoW is all zero.", W_index)
    #  word_temp = raw_input("Please word?>")
    #  Otb_B = [int(W_index[i] == word_temp) * N_best for i in range(len(W_index))]
    #  print("BoW (NEW):",  Otb_B)

    #########
    #Path-Planning
    Path, Path_ROS, PathWeightMap, Path_one = path_calculate.PathPlanner(Otb_B, start_inv, THETA, CostMapProb, outputfile, speech_num, outputname) #gridmap, costmap)

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
    save_data.SavePath(start, [], Path, Path_ROS, outputname)


    #Save the PathWeightMap(PathPlanner内部で実行)
    #####save_data.SaveProbMap(PathWeightMap, outputname)
    
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
    
    
    #すべてのステップにおけるlog likelihoodの値を保存
    save_data.SaveLogLikelihood(LogLikelihood_step,0,0,outputname)
    
    #すべてのステップにおける累積報酬（sum log likelihood）の値を保存
    save_data.SaveLogLikelihood(LogLikelihood_sum,1,0,outputname)
    
    
    print("[END] SpCoNavi. (Viterbi)")
