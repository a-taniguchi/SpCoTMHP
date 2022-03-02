#coding:utf-8

###########################################################
# SpCoTMHPi: Spatial Concept-based Path-Planning Program for SIGVerse
# Path-Planning Program by A star algorithm (ver. approximate inference)
# Path Selection: minimum cost (- log-likelihood) in a path trajectory
# Akira Taniguchi 2022/02/07
# Spacial Thanks: Ryo Ozaki, Shuya Ito
###########################################################

##Command: 
#python3 spcotmhp_astar_metric.py trialname mapname iteration sample type_gauss
#python3 spcotmhp_astar_metric.py 3LDK_01 s3LDK_01 1 0 g

import sys
import time
import numpy as np
#import scipy as sp
from scipy.stats import multivariate_normal,multinomial
import matplotlib.pyplot as plt
import spconavi_read_data
import spconavi_save_data
#import spconavi_viterbi_path_calculate as spconavi_viterbi_path_calculate
from __init__ import *
from submodules import *

tools     = spconavi_read_data.Tools()
read_data = spconavi_read_data.ReadingData()
save_data = spconavi_save_data.SavingData()
#path_calculate = spconavi_viterbi_path_calculate.PathPlanner()

#Definition of action (functions in spconavi_read_data)
action_functions = [tools.right, tools.left, tools.up, tools.down, tools.stay] #, migiue, hidariue, migisita, hidarisita]
cost_of_actions  = np.log( np.ones(len(action_functions)) / float(len(action_functions)) ) #[    1/5,    1/5,  1/5,    1/5,    1/5]) #, ,    1,        1,        1,          1]


"""
#GaussMap make (no use) #Ito
def PostProb_ij(Index_temp,Mu,Sig,map_length,map_width, CostMapProb,it):
        if (CostMapProb[Index_temp[1]][Index_temp[0]] != 0.0): 
            X_temp = tools.Array_index_To_Map_coordinates(Index_temp)  #map と縦横の座標系の軸が合っているか要確認
            #print X_temp,Mu
            sum_i_GaussMulti = [ multivariate_normal.pdf(X_temp, mean=Mu[it], cov=Sig[it])] ##########np.array( ) !!! np.arrayにすると, numbaがエラーを吐く
            PostProb = np.sum(sum_i_GaussMulti)  #sum_c_ProbCtsum_i
        else:
            PostProb = 0.0
        return PostProb
"""

#GaussMap make (use) #Ito
def PostProbMap_Gauss(CostMapProb,Mu,Sig,map_length,map_width,it): #,IndexMap):
        x,y = np.meshgrid(np.linspace(-10.0,9.92,map_width),np.linspace(-10.0,9.92,map_length))
        pos = np.dstack((x,y))    
        #PostProbMap = np.array([ [ PostProb_ij([width, length],Mu,Sig,map_length,map_width, CostMapProb,it) for width in xrange(map_width) ] for length in xrange(map_length) ])
        PostProb=multivariate_normal(Mu[it],Sig[it]).pdf(pos)

        return CostMapProb * PostProb


#GaussMap make (use) #Ito->Akira
def PostProbMap_NormalizedGauss(CostMapProb,Mu,Sig,map_length,map_width,it): #,IndexMap):
        x,y = np.meshgrid(np.linspace(-10.0,9.92,map_width),np.linspace(-10.0,9.92,map_length))
        pos = np.dstack((x,y))    
        #PostProbMap = np.array([ [ PostProb_ij([width, length],Mu,Sig,map_length,map_width, CostMapProb,it) for width in xrange(map_width) ] for length in xrange(map_length) ])
        bunbo = np.sum([ multivariate_normal(Mu[k],Sig[k]).pdf(pos) for k in range(len(Mu)) ], 0)
        PostProb = multivariate_normal(Mu[it],Sig[it]).pdf(pos) / bunbo

        return CostMapProb * PostProb



###↓### Sampling of goal candidates ############################################
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
        goal_candidate += [tools.Map_coordinates_To_Array_index(multivariate_normal.rvs(mean=Myu[it], cov=S[it], size=1, random_state=None))]
        count_it += 1
  #Xt_max = Map_coordinates_To_Array_index( [ Xp[pox.index(max(pox))][0], Xp[pox.index(max(pox))][1] ] ) #[0.0,0.0] ##確率最大の座標候補
  goal_candidate_tuple = [(goal_candidate[j][1], goal_candidate[j][0]) for j in range(Sampling_J)]
  print("Goal candidates:", goal_candidate_tuple)
  return goal_candidate_tuple
###↑### Sampling of goal candidates ############################################



### A star algorithm (by Ryo Ozaki) ############################################
def a_star(start, goal, maze, action_functions, cost_of_actions, PathWeightMap):
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
    open_list_key.append(0 + tools.Manhattan_distance(start, goal))
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
            q_pev = tools.Manhattan_distance(q, goal) * np.log(float(len(action_functions))) #heuristic function
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
    return Path, p_cost
### A star algorithm (by Ryo Ozaki) ############################################


#################################################
print("[START] SpCoTMHP. (A star metric path)")

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
outputfile = filename + navigation_folder #+ "astar_node/" #outputfolder + trialname + navigation_folder #Ito
if (type_gauss == "g"):
    outputsubfolder = outputfile + "astar_node_gauss/"
else:
    outputsubfolder = outputfile + "astar_node/"
#"T"+str(T_horizon)+"N"+str(N_best)+"A"+str(Approx)+"S"+str(init_position_num)+"G"+str(speech_num)

Makedir( outputfile )
Makedir( outputsubfolder )


#Read the files of learned parameters  #THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
THETA = read_data.ReadParameters(iteration, sample, filename, trialname)
W, W_index, Mu, Sig, pi, phi_l, K, L = THETA
#W_index = THETA[1]

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



##Read the map file
gridmap = read_data.ReadMap(outputfile)
map_length, map_width = gridmap.shape
#GridMapProb = 1*(gridmap == 0)

##Read the cost map file
#CostMap = read_data.ReadCostMap(outputfile)
#CostMapProb_tmp = (100.0 - CostMap)/100
#CostMapProb = CostMapProb_tmp * GridMapProb

#Read the probabilistic cost map file
CostMapProb = read_data.ReadCostMapProb(outputfile)


for st_i in range(K):
 for gl_i in range(K):
  if st_i == gl_i:
       Distance_save[st_i][gl_i]=0
       Like_save[st_i][gl_i]=0
  elif psi[st_i][gl_i] == 1:
    St=st_i
    Gl=gl_i

    outputname = outputsubfolder + "Astar_SpCoTMHP_"+"S"+str(St)+"_G"+str(Gl)
    if (type_gauss == "g"):
       PathWeightMap = PostProbMap_Gauss(CostMapProb,Mu,Sig,map_length,map_width,Gl)
    else:
       PathWeightMap = PostProbMap_NormalizedGauss(CostMapProb,Mu,Sig,map_length,map_width,Gl)
    
    

    #####描画
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
    #goal_candidate = Sampling_goal(Otb_B, THETA) #(0,0)

    # スタート：ガウス平均
    ss=tools.Map_coordinates_To_Array_index(Mu[St])
    start=(ss[1],ss[0]) #スタート位置を指定

    # ゴール：ガウス平均
    gg=tools.Map_coordinates_To_Array_index(Mu[Gl])
    #print(gg[0])

    goal_candidate = [[gg[1],gg[0]]]

    #J = Sampling_J #len(goal_candidate)
    #if(J != THETA[6]):
    # print("[WARNING] J is not K",J,K)
    J=1
    p_cost_candidate = [0.0 for j in range(J)]
    Path_candidate   = [[0.0] for j in range(J)]
    Like_candidate   = [0.0 for j in range(J)]
    #print(goal_candidate)

    if (SAVE_time == 1):
        #Substitution of start time
        start_time = time.time()

    ###goal候補ごとにA*を実行
    for gc_index in range(J):
        goal = goal_candidate[gc_index]
        Path, p_cost = a_star(start, goal, gridmap, action_functions, cost_of_actions, PathWeightMap)    

        Like_candidate[gc_index] = p_cost

        p_cost_candidate[gc_index] = p_cost / float(len(Path))
        Path_candidate[gc_index]   = Path    


    ### select the goal of expected cost
    expect_gc_index = np.argmin(p_cost_candidate)
    Path = Path_candidate[expect_gc_index]
    goal = goal_candidate[expect_gc_index]
    print("Goal:", goal)
    

    if (SAVE_time == 1):
        #PP終了時刻を保持
        end_pp_time = time.time()
        time_pp = end_pp_time - start_time 
        fp = open( outputname + "_time_pp.txt", 'w')
        fp.write(str(time_pp)+"\n")
        fp.close()
        
    for i in range(len(Path)):
      plt.plot(Path[i][0], Path[i][1], "s", color="tab:red", markersize=1)


    #The moving distance of the path
    Distance = tools.PathDistance(Path)
    Distance_save[st_i][gl_i] = Distance
    Like_save[st_i][gl_i]=Like_candidate[expect_gc_index] # 実際は尤度ではなくA*のコスト値

    #Save the moving distance of the path
    save_data.SavePathDistance(Distance, outputname)
    print("Path distance using A* algorithm is "+ str(Distance))

    #計算上パスのx,yが逆になっているので直す
    Path_inv = [[Path[t][1], Path[t][0]] for t in range(len(Path))]
    Path_inv.reverse()
    Path_ROS = Path_inv #使わないので暫定的な措置

    #Save the path
    save_data.SavePath(start, [goal[1], goal[0]], Path_inv, Path_ROS, outputname)


    #Read the emission probability file 
    #PathWeightMap = ReadProbMap(outputfile)


    #Save the log-likelihood of the path
    #PathWeightMapとPathからlog likelihoodの値を再計算する
    LogLikelihood_step = np.zeros(T_horizon)
    LogLikelihood_sum  = np.zeros(T_horizon)

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
    save_data.SaveLogLikelihood(LogLikelihood_step,0,0,outputname)

    #すべてのステップにおける累積報酬（sum log likelihood）の値を保存
    save_data.SaveLogLikelihood(LogLikelihood_sum,1,0,outputname)
    


    #Save path trajectory in the map as a color image
    plt.savefig(outputname + '_Path.png', dpi=300)#, transparent=True
    plt.savefig(outputname + '_Path.pdf', dpi=300)#, transparent=True
    plt.clf()
  else:
    Distance_save[st_i][gl_i]=0
    Like_save[st_i][gl_i]=0 

print("[END] SpCoTMHP. (A star metric path)")

#if (type_gauss == "g"):
#    outputsubfolder = outputfile + "Astar_SpCoTMHP_gauss_"
#else:
outputsubfolder = outputfile + "Astar_SpCoTMHP_"

np.savetxt(outputsubfolder+"distance.csv",Distance_save,delimiter=",")
np.savetxt(outputsubfolder+"cost.csv",Like_save,delimiter=",")
