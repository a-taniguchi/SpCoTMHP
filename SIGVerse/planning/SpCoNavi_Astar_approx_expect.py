#coding:utf-8

###########################################################
# SpCoNavi: Spatial Concept-based Path-Planning Program for SIGVerse
# Path-Planning Program by A star algorithm (ver. approximate inference)
# Path Selection: expected log-likelihood per pixel in a path trajectory
# Akira Taniguchi 2019/06/24-2019/07/02-2019/09/20-2019/10/26-
# Spacial Thanks: Ryo Ozaki
###########################################################

##Command: 
#python ./SpCoNavi_Astar_approx.py trialname mapname iteration sample init_position_num speech_num
#python ./SpCoNavi_Astar_approx.py 3LDK_01 s1DK_01 1 0 0 0

import sys
import random
import string
import time
import numpy as np
import scipy as sp
from numpy.linalg import inv, cholesky
from scipy.stats import chi2,multivariate_normal,multinomial
from math import pi as PI
from math import cos,sin,sqrt,exp,log,fabs,fsum,degrees,radians,atan2
import matplotlib.pyplot as plt
import collections
from __init__ import *
#from submodules import *

def right(pos):
    return (pos[0], pos[1] + 1)

def left(pos):
    return (pos[0], pos[1] - 1)

def up(pos):
    return (pos[0] - 1, pos[1])

def down(pos):
    return (pos[0] + 1, pos[1])

def stay(pos):
    return (pos[0], pos[1])

def Manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

#Read the map data⇒2-dimension array に格納
def ReadMap(outputfile):
    #outputfolder + trialname + navigation_folder + map.csv
    gridmap = np.loadtxt(outputfile + "map.csv", delimiter=",")
    print("Read map: " + outputfile + "map.csv")
    return gridmap

#Read the cost map data⇒2-dimension array に格納
def ReadCostMap(outputfile):
    #outputfolder + trialname + navigation_folder + contmap.csv
    costmap = np.loadtxt(outputfile + "costmap.csv", delimiter=",")
    print("Read costmap: " + outputfile + "contmap.csv")
    return costmap

#Load the probability cost map used for path calculation
def ReadCostMapProb(outputfile):
    # Read the result from the file
    output = outputfile + "CostMapProb.csv"
    CostMapProb = np.loadtxt(output, delimiter=",")
    print("Read CostMapProb: " + output)
    return CostMapProb

#Load the probability value map used for path calculation
def ReadProbMap(outputfile):
    # Read the result from the file
    output = outputfile + "N"+str(N_best)+"G"+str(speech_num) + "_PathWeightMap.csv"
    PathWeightMap = np.loadtxt(output, delimiter=",")
    print( "Read PathWeightMap: " + output)
    return PathWeightMap


#Save the path trajectory
def SavePath(X_init, X_goal, Path, Path_ROS, outputname):
    print("PathSave")
    if (SAVE_X_init == 1):
      # Save robot initial position and goal as file (index)
      np.savetxt(outputname + "_X_init.csv", X_init, delimiter=",")
      np.savetxt(outputname + "_X_goal.csv", X_goal, delimiter=",")
      # Save robot initial position and goal as file (ROS)
      np.savetxt(outputname + "_X_init_ROS.csv", Array_index_To_Map_coordinates(X_init), delimiter=",")
      np.savetxt(outputname + "_X_goal_ROS.csv", Array_index_To_Map_coordinates(X_goal), delimiter=",")

    # Save the result to the file (index)
    np.savetxt(outputname + "_Path.csv", Path, delimiter=",")
    # Save the result to the file (ROS)
    np.savetxt(outputname + "_Path_ROS.csv", Path_ROS, delimiter=",")
    print("Save Path: " + outputname + "_Path.csv and _Path_ROS.csv")


#Save the log likelihood for each time-step
def SaveLogLikelihood(outputname, LogLikelihood,flag,flag2):
    # Save the result to the file 
    if (flag2 == 0):
      if   (flag == 0):
        output_likelihood = outputname + "_Log_likelihood_step.csv"
      elif (flag == 1):
        output_likelihood = outputname + "_Log_likelihood_sum.csv"
    else:
      if   (flag == 0):
        output_likelihood = outputname + "_Log_likelihood_step" + str(flag2) + ".csv"
      elif (flag == 1):
        output_likelihood = outputname + "_Log_likelihood_sum" + str(flag2) + ".csv"

    np.savetxt( output_likelihood, LogLikelihood, delimiter=",")
    print("Save LogLikekihood: " + output_likelihood)

#ROSのmap 座標系をPython内の2-dimension array のインデックス番号に対応付ける
def Map_coordinates_To_Array_index(X):
    X = np.array(X)
    Index = np.round( (X - origin) / resolution ).astype(int) #四捨五入してint型にする
    return Index

#Python内の2-dimension array のインデックス番号からROSのmap 座標系への変換
def Array_index_To_Map_coordinates(Index):
    Index = np.array(Index)
    X = np.array( (Index * resolution) + origin )
    return X

#The moving distance of the pathを計算する
def PathDistance(Path):
    Distance = len(collections.Counter(Path))
    print("Path Distance is ", Distance)
    return Distance

#Save the moving distance of the path
def SavePathDistance(Distance):
    # Save the result to the file 
    output = outputname + "_Distance.csv"
    np.savetxt( output, np.array([Distance]), delimiter=",")
    print("Save Distance: " + output)

#Save the moving distance of the path
def SavePathDistance_temp(Distance,temp):
    # Save the result to the file 
    output = outputname + "_Distance"+str(temp)+".csv"
    np.savetxt( output, np.array([Distance]), delimiter=",")
    print("Save Distance: " + output)

#Read the parameters of learned spatial concepts
def ReadParameters(iteration, sample, filename, trialname):
    #THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
    #r = iteration
    """
    i = 0
    for line in open(filename + 'index' + str(r) + '.csv', 'r'):   ##読み込む
        itemList = line[:-1].split(',')
        #print itemList
        if (i == 0):
          L = len(itemList) -1
        elif (i == 1):
          K = len(itemList) -1
        i += 1
    print "L:",L,"K:",K
    """

    W_index = []
    i = 0
    #Read text file
    for line in open(filename + "/" + trialname + '_w_index_' + str(iteration) + '_' + str(sample) + '.csv', 'r'): 
        itemList = line[:-1].split(',')
        if(i == 1):
            for j in range(len(itemList)):
              if (itemList[j] != ""):
                W_index = W_index + [itemList[j]]
        i = i + 1
    
    #####parameter W, μ, Σ, φ, πを入力する#####
    Mu    = [ np.array([ 0.0, 0.0 ]) for i in range(K) ]  #[ np.array([[ 0.0 ],[ 0.0 ]]) for i in range(K) ]      #the position distribution (Gaussian)の平均(x,y)[K]
    Sig   = [ np.array([ [0.0, 0.0],[0.0, 0.0] ]) for i in range(K) ]      #the position distribution (Gaussian)の共分散(2×2-dimension)[K]
    W     = [ [0.0 for j in range(len(W_index))] for c in range(L) ]  #the name of place(multinomial distribution: W_index-dimension)[L]
    #theta = [ [0.0 for j in range(DimImg)] for c in range(L) ] 
    Pi    = [ 0.0 for c in range(L)]     #index of spatial conceptのmultinomial distribution(L-dimension)
    Phi_l = [ [0.0 for i in range(K)] for c in range(L) ]  #index of position distributionのmultinomial distribution(K-dimension)[L]
      
    i = 0
    ##Mu is read from the file
    for line in open(filename + "/" + trialname + '_Myu_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        #Mu[i] = np.array([ float(itemList[0]) - origin[0] , float(itemList[1]) - origin[1] ]) / resolution
        Mu[i] = np.array([ float(itemList[0]) , float(itemList[1]) ])
        i = i + 1
      
    i = 0
    ##Sig is read from the file
    for line in open(filename + "/" + trialname + '_S_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        #Sig[i] = np.array([[ float(itemList[0])/ resolution, float(itemList[1]) ], [ float(itemList[2]), float(itemList[3])/ resolution ]]) #/ resolution
        Sig[i] = np.array([[ float(itemList[0]), float(itemList[1]) ], [ float(itemList[2]), float(itemList[3]) ]]) 
        i = i + 1
      
    ##phi is read from the file
    c = 0
    #Read text file
    for line in open(filename + "/" + trialname + '_phi_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in range(len(itemList)):
            if itemList[i] != "":
              Phi_l[c][i] = float(itemList[i])
        c = c + 1
        
    ##Pi is read from the file
    for line in open(filename + "/" + trialname + '_pi_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in range(len(itemList)):
          if itemList[i] != '':
            Pi[i] = float(itemList[i])
      
    ##W is read from the file
    c = 0
    #Read text file
    for line in open(filename + "/" + trialname + '_W_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in range(len(itemList)):
            if itemList[i] != '':
              #print c,i,itemList[i]
              W[c][i] = float(itemList[i])
        c = c + 1

    """
    ##theta is read from the file
    c = 0
    #Read text file
    for line in open(filename + 'theta' + str(r) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in range(len(itemList)):
            if itemList[i] != '':
              #print c,i,itemList[i]
              theta[c][i] = float(itemList[i])
        c = c + 1
    """

    THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
    return THETA


###↓###ゴール候補のサンプリング############################################
def Sampling_goal(Otb_B, THETA):
  #THETAを展開
  W, W_index, Myu, S, pi, phi_l, K, L = THETA
  
  #Prob math func of p(it | Otb_B, THETA) = Σc p(it | phi_c)p(st=Otb_B | Wc)p(c | pi)
  pmf_it = np.ones(K)
  for i in range(K):
      sum_Ct = np.sum([phi_l[c][i] * multinomial.pmf(Otb_B, sum(Otb_B), W[c]) * pi[c] for c in range(L)])
      pmf_it[i] = sum_Ct

  #Normalization
  pmf_it_n = np.array([pmf_it[i] / float(np.sum(pmf_it)) for i in range(K)])

  #Sampling it from multinomial distribution
  sample_it = multinomial.rvs(Sampling_J, pmf_it_n, size=1, random_state=None)
  print(sample_it)
  goal_candidate = []
  for it in range(K):
      count_it = 0
      while (count_it < sample_it[0][it]):
        goal_candidate += [Map_coordinates_To_Array_index(multivariate_normal.rvs(mean=Myu[it], cov=S[it], size=1, random_state=None))]
        count_it += 1
  #Xt_max = Map_coordinates_To_Array_index( [ Xp[pox.index(max(pox))][0], Xp[pox.index(max(pox))][1] ] ) #[0.0,0.0] ##確率最大の座標候補
  goal_candidate_tuple = [(goal_candidate[j][1], goal_candidate[j][0]) for j in range(Sampling_J)]
  print("Goal candidates:", goal_candidate_tuple)
  return goal_candidate_tuple
###↑###ゴール候補のサンプリング############################################



#################################################
print("[START] SpCoNavi by A star algorithm.")

#map dataの入った部屋環境folder name（学習済みparameter folder name） is requested
trialname = sys.argv[1]

#map file name is requested
mapname = sys.argv[2]

#iteration is requested
iteration = sys.argv[3] #1

#sample is requested
sample = sys.argv[4] #0

#robot initial positionの候補番号 is requested
init_position_num = sys.argv[5] #0

#the file number for speech instruction is requested   
speech_num = sys.argv[6] #0

if (SAVE_time == 1):
    #開始時刻を保持
    start_time = time.time()

start_list = [0, 0] #Start_Position[int(init_position_num)]#(83,39) #(92,126) #(126,92) #(1, 1)
start_list[0] = int(sys.argv[7]) #0
start_list[1] = int(sys.argv[8]) #0
start = (start_list[0], start_list[1])
print("Start:", start)
#goal  = (95,41) #(97,55) #(55,97) #(height-2, width-2)

##FullPath of folder
filename = outputfolder_SIG + trialname #+ "/" 
print(filename, iteration, sample)
outputfile = filename + navigation_folder #outputfolder + trialname + navigation_folder
#outputname = outputfile + "Astar_SpCo_"+"N"+str(N_best)+"A"+str(Approx)+"S"+str(init_position_num)+"G"+str(speech_num)
outputname = outputfile + "Astar_Approx_expect_"+"N"+str(N_best)+"A"+str(Approx)+"S"+str(start)+"G"+str(speech_num)

#"T"+str(T_horizon)+"N"+str(N_best)+"A"+str(Approx)+"S"+str(init_position_num)+"G"+str(speech_num)
#maze_file = outputfile + mapname + ".pgm"
#maze_file = "../CoRL/data/1DK_01/navi/s1DK_01.pgm" #"./sankou/sample_maze.txt"

#maze = np.loadtxt(maze_file, dtype=int)
#height, width = maze.shape
"""
##########
#Read the image PGM file
#http://www.not-enough.org/abe/manual/api-aa09/fileio2.html
infile = open(maze_file , 'rb') #sys.argv[1]

for i in range(4): #最初の4行は無視
    d = infile.readline()
    print(d[:-1])
    if (i == 2): #3行目を読み込む
        item   = str(d[:-1]).split(' ')
        #print(item)
        height = int((item[0][2:]))
        width  = int((item[1][0:-1]))

maze = np.zeros((height, width))
print(height, width)

for h in range(height):
    for w in range(width):
        d = infile.read(1)
        maze[h][w] = int(255 - ord(d))/255

infile.close
##########
"""
maze = ReadMap(outputfile)
height, width = maze.shape

action_functions = [right, left, up, down, stay] #, migiue, hidariue, migisita, hidarisita]
cost_of_actions  = np.log( np.ones(len(action_functions)) / float(len(action_functions)) ) #[    1/5,    1/5,  1/5,    1/5,    1/5]) #, ,    1,        1,        1,          1]

#Read the files of learned parameters  #THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
THETA = ReadParameters(iteration, sample, filename, trialname)
W_index = THETA[1]

#####Estimate the goal point by spatial concept
Otb_B = [int(W_index[i] == Goal_Word[int(speech_num)]) * N_best for i in range(len(W_index))]
print("BoW:", Otb_B)

#Path-Planning
#Path, Path_ROS, PathWeightMap, Path_one = PathPlanner(Otb_B, Start_Position[int(init_position_num)], THETA, CostMapProb) #gridmap, costmap)

#Read the emission probability file 
PathWeightMap = ReadProbMap(outputfile)

#####描画
#plt.imshow(maze, cmap="binary")
gridmap = maze
plt.imshow(gridmap + (40+1)*(gridmap == -1), origin='lower', cmap='binary', vmin = 0, vmax = 100, interpolation='none') #, vmin = 0.0, vmax = 1.0)
     
plt.xticks(rotation=90)
plt.tick_params(axis='x', which='major', labelsize=8)
plt.tick_params(axis='y', which='major', labelsize=8)
#plt.xlim([380,800])             #x軸の範囲
#plt.ylim([180,510])             #y軸の範囲
plt.xlabel('X', fontsize=10)
plt.ylabel('Y', fontsize=10)
#plt.xticks(np.arange(width), np.arange(width))
#plt.yticks(np.arange(height), np.arange(height))
plt.gca().set_aspect('equal')

# スタートとゴールをプロットする
#plt.plot(start[1], start[0], "D", color="tab:blue", markersize=1)
#plt.plot(goal[1], goal[0], "D", color="tab:pink", markersize=1)

#plt.show()

###goalの候補を複数個用意する
goal_candidate = Sampling_goal(Otb_B, THETA) #(0,0)
J = len(goal_candidate)
if(J != THETA[6]):
    print("[WARNING] J is not K",J,K)
p_cost_candidate = [0.0 for j in range(J)]
Path_candidate = [[0.0] for j in range(J)]

###goal候補ごとにA*を実行
for gc_index in range(J):
    goal = goal_candidate[gc_index]
    if (maze[goal[0]][goal[1]] != 0):
        print("[ERROR] goal",maze[goal[0]][goal[1]],"is not 0.")

    ###START A*
    open_list = []
    open_list_cost = []
    open_list_key = []
    closed_list = []
    closed_list_cost = []
    closed_list_key = []
    open_list.append(start)
    open_list_cost.append(0)
    open_list_key.append(0 + Manhattan_distance(start, goal))
    OYA = {}
    ko = (0), (0)
    Path = []

    while open_list:
        sorted_idx = np.argsort(open_list_key, kind="stable")
        pop_idx = sorted_idx[0]
        p = open_list.pop(pop_idx)
        p_cost = open_list_cost.pop(pop_idx)
        p_key = open_list_key.pop(pop_idx)
        closed_list.append(p)
        closed_list_cost.append(p_cost)
        closed_list_key.append(p_key)
        if p == goal:
            break
        for act_func, act_cost in zip(action_functions, cost_of_actions):
            q = act_func(p)
            if (int(maze[q]) != 0):
                continue
            q_cost = p_cost - act_cost - np.log(PathWeightMap[q[0]][q[1]])  #current sum cost and action cost 
            q_pev = Manhattan_distance(q, goal) * np.log(float(len(action_functions))) #heuristic function
            q_key = q_cost - q_pev

            if q in open_list:
                idx = open_list.index(q)
                key = open_list_key[idx]
                if key > q_key:
                    open_list_key[idx] = q_key
                    open_list_cost[idx] = q_cost
            elif q in closed_list:
                idx = closed_list.index(q)
                key = closed_list_key[idx]
                if key > q_key:
                    closed_list.pop(idx)
                    closed_list_cost.pop(idx)
                    closed_list_key.pop(idx)
                    open_list.append(q)
                    open_list_cost.append(q_cost)
                    open_list_key.append(q_key)
                    #plt.quiver(p[1], p[0], (q[1]-p[1]), (q[0]-p[0]), angles='xy', scale_units='xy', scale=1, color="tab:red")
                    OYA[(q[1], q[0])] = (p[1], p[0])
                    ko = (q[1]), (q[0])
                    #print(ko)
            else:
                open_list.append(q)
                open_list_cost.append(q_cost)
                open_list_key.append(q_key)
                #plt.quiver(p[1], p[0], (q[1]-p[1]), (q[0]-p[0]), angles='xy', scale_units='xy', scale=1, color="tab:red")
                OYA[(q[1], q[0])] = (p[1], p[0])
                ko = (q[1]), (q[0])
                #print(ko)

    #最適経路の決定: ゴールから親ノード（どこから来たか）を順次たどっていく
    #i = len(OYA)
    #for oyako in reversed(OYA):
    ko_origin = ko
    ko = (goal[1], goal[0])
    print(ko,goal)
    #for i in range(p_cost):
    while(ko != (start[1],start[0])):
        #print(OYA[ko])
        try:
            Path = Path + [OYA[ko]]
        except KeyError:
            ko = ko_origin
            Path = Path + [OYA[ko]]
            print("NOT END GOAL.")
        
        ko = OYA[ko]
        #i = len(Path)
        #print(i, ko)
        #i -= 1

    print(goal,": Total cost using A* algorithm is "+ str(p_cost))
    p_cost_candidate[gc_index] = p_cost / float(len(Path))
    Path_candidate[gc_index] = Path    

    """
    #PathWeightMapとPathからlog likelihoodの値を再計算する
    LogLikelihood_step = np.zeros(T_horizon)
    LogLikelihood_sum = np.zeros(T_horizon)

    for i in range(T_horizon):
        if (i < len(Path)):
            t = i
        else:
            t = len(Path) -1
        #print PathWeightMap.shape, Path[t][0], Path[t][1]
        LogLikelihood_step[i] = np.log(PathWeightMap[ Path_inv[t][0] ][ Path_inv[t][1] ])
        if (t == 0):
            LogLikelihood_sum[i] = LogLikelihood_step[i]
        elif (t >= 1):
            LogLikelihood_sum[i] = LogLikelihood_sum[i-1] + LogLikelihood_step[i]
    
    LogLikelihood_step_candidate[gc_index] = LogLikelihood_step
    LogLikelihood_sum_candidate[gc_index] = LogLikelihood_sum
    """

### select the goal of expected cost
expect_gc_index = np.argmin(p_cost_candidate)
Path = Path_candidate[expect_gc_index]
goal = goal_candidate[expect_gc_index]
print("Goal:", goal)
#LogLikelihood_step
#LogLikelihood_sum

if (SAVE_time == 1):
    #PP終了時刻を保持
    end_pp_time = time.time()
    time_pp = end_pp_time - start_time #end_recog_time
    fp = open( outputname + "_time_pp.txt", 'w')
    fp.write(str(time_pp)+"\n")
    fp.close()
    
for i in range(len(Path)):
  plt.plot(Path[i][0], Path[i][1], "s", color="tab:red", markersize=1)


#The moving distance of the path
Distance = PathDistance(Path)

#Save the moving distance of the path
SavePathDistance(Distance)

print("Path distance using A* algorithm is "+ str(Distance))

#計算上パスのx,yが逆になっているので直す
Path_inv = [[Path[t][1], Path[t][0]] for t in range(len(Path))]
Path_inv.reverse()
Path_ROS = Path_inv #使わないので暫定的な措置
#パスを保存
SavePath(start, [goal[1], goal[0]], Path_inv, Path_ROS, outputname)


#Read the emission probability file 
#PathWeightMap = ReadProbMap(outputfile)

#Save the log-likelihood of the path
#PathWeightMapとPathからlog likelihoodの値を再計算する
LogLikelihood_step = np.zeros(T_horizon)
LogLikelihood_sum = np.zeros(T_horizon)

for i in range(T_horizon):
    if (i < len(Path)):
        t = i
    else:
        t = len(Path) -1
    #print PathWeightMap.shape, Path[t][0], Path[t][1]
    LogLikelihood_step[i] = np.log(PathWeightMap[ Path_inv[t][0] ][ Path_inv[t][1] ])
    if (t == 0):
        LogLikelihood_sum[i] = LogLikelihood_step[i]
    elif (t >= 1):
        LogLikelihood_sum[i] = LogLikelihood_sum[i-1] + LogLikelihood_step[i]


#すべてのステップにおけるlog likelihoodの値を保存
SaveLogLikelihood(outputname, LogLikelihood_step,0,0)

#すべてのステップにおける累積報酬（sum log likelihood）の値を保存
SaveLogLikelihood(outputname, LogLikelihood_sum,1,0)



#plt.show()

#Save path trajectory in the map as a color image
#output = outputfile + "N"+str(N_best)+"G"+str(speech_num)
plt.savefig(outputname + '_Path.png', dpi=300)#, transparent=True
#plt.savefig(outputfile + "step/" + conditions + '_Path_Weight' +  str(temp).zfill(3) + '.png', dpi=300)#, transparent=True
plt.savefig(outputname + '_Path.pdf', dpi=300)#, transparent=True
plt.clf()

