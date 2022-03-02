#coding:utf-8

###########################################################
# SpCoNavi: Spatial Concept-based Path-Planning Program for SIGVerse
# Path-Planning Program by A star algorithm (ver. approximate inference)
# Path Selection: expected log-likelihood per pixel in a path trajectory
# Akira Taniguchi 2022/02/07-2022/02/25
# Spacial Thanks: Ryo Ozaki, Shoichi Hasegawa
###########################################################

##Command: 
#python3 albert_b_spconavi_astar_execute.py trialname mapname iteration sample init_position_num speech_num initial_position_x initial_position_y waypoint_word
#python3 albert_b_spconavi_astar_execute.py albertGMM01 map 1 9 init 0 590 340 -1

import os
import sys
import time
import numpy as np
from scipy.stats import multivariate_normal,multinomial
import matplotlib.pyplot as plt
from PIL import Image,ImageOps #, ImageDraw
from __init__ import *
from submodules import *
import spconavi_read_data
import spconavi_save_data
import albert_b_spconavi_viterbi_path_calculate as albert_b_spconavi_viterbi_path_calculate

tools     = spconavi_read_data.Tools()
read_data = spconavi_read_data.ReadingData()
save_data = spconavi_save_data.SavingData()
path_calculate = albert_b_spconavi_viterbi_path_calculate.PathPlanner()

#Definition of action (functions in spconavi_read_data)
action_functions = [tools.right, tools.left, tools.up, tools.down, tools.stay] #, migiue, hidariue, migisita, hidarisita]
cost_of_actions  = np.log( np.ones(len(action_functions)) / float(len(action_functions)) ) #[    1/5,    1/5,  1/5,    1/5,    1/5]) #, ,    1,        1,        1,          1]


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
        mu_ai = tools.Map_coordinates_To_Array_index_albert(Myu[it])
        if (mu_ai[1] <= 800) and (mu_ai[0] <= 1184):
            sample = tools.Map_coordinates_To_Array_index_albert(multivariate_normal.rvs(mean=Myu[it], cov=S[it], size=1, random_state=None))
        else:
            sample = tools.Map_coordinates_To_Array_index_albert(multivariate_normal.rvs(mean=[1184/2.0,800/2.0], cov=np.eye(2), size=1, random_state=None))
        if (0 < sample[1] <= 800) and (0 < sample[0] <= 1184):
            goal_candidate += [sample]
            count_it += 1
  #Xt_max = Map_coordinates_To_Array_index_albert( [ Xp[pox.index(max(pox))][0], Xp[pox.index(max(pox))][1] ] ) #[0.0,0.0] ##確率最大の座標候補
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
print("[START] SpCoNavi by A star approx. algorithm.")

#map dataの入った部屋環境folder name（学習済みparameter folder name） is requested
trialname = sys.argv[1]

#map file name is requested
#mapname = sys.argv[2]

#iteration is requested
iteration = sys.argv[3] #1

#sample is requested
sample = sys.argv[4] #0

#robot initial positionの候補番号 is requested
#init_position_num = sys.argv[5] #0

#the file number for speech instruction is requested   
speech_num = sys.argv[6] #0


#start_list = [0, 0] #Start_Position[int(init_position_num)]
#start_list[0] = int(sys.argv[7]) #0
#start_list[1] = int(sys.argv[8]) #0
start = (int(sys.argv[7]), int(sys.argv[8]))
start_inv = [start[1],start[0]]
print("Start:", start)


#中間地点の単語番号を指定 (未実装：複数指定の場合、コンマ区切りする)
waypoint_word = sys.argv[9] # -1:中間なし


if (waypoint_word != ""):
    WP_list    = waypoint_word[:].split(',')
    print("WP:", WP_list)

    if (int(WP_list[0]) == -1):
        tyukan = 0
    else:
        tyukan = 1
else:
    tyukan = 0


if (SAVE_time == 1):
    #開始時刻を保持
    start_time = time.time()

##FullPath of folder
filename = outputfolder_albert + trialname #+ "/" 
print(filename, iteration, sample)
outputfile = filename + navigation_folder #outputfolder + trialname + navigation_folder
outputsubfolder = outputfile + "spconavi_astar_min/"
if (tyukan == 0):
    outputname = outputsubfolder + "J"+str(Sampling_J)+"N"+str(N_best)+"A"+str(Approx)+"S"+str(start)+"G"+str(speech_num)+"/"
elif (tyukan == 1):
    outputname = outputsubfolder + "J"+str(Sampling_J)+"N"+str(N_best)+"A"+str(Approx)+"S"+str(start)+"H"+str(waypoint_word)+"G"+str(speech_num)+"/"
#"T"+str(T_horizon)+"N"+str(N_best)+"A"+str(Approx)+"S"+str(init_position_num)+"G"+str(speech_num)

Makedir( outputfile )
Makedir( outputsubfolder )
Makedir( outputname )

gridmap = read_data.ReadMap(outputfile)
height, width = gridmap.shape


# map の .pgm file を読み込み
map_file_path = inputfolder_albert + map_file + '.pgm' #roslib.packages.get_pkg_dir('em_spco_ae') + '/map/' + self.map_file + '/map.pgm'
map_image     = Image.open(map_file_path)
map_image     = ImageOps.flip(map_image)      # 上下反転

# height and width を得る
#width, height = map_image.size
#print(map_image.size)

#Read the probabilistic cost map file
CostMapProb = read_data.ReadCostMapProb(outputfile)


#Read the files of learned parameters  #THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
THETA = read_data.ReadParameters(iteration, sample, filename, trialname)
W_index = THETA[1]

#####Estimate the goal point by spatial concept
Otb_B = [int(W_index[i] == Goal_Word_albert[int(speech_num)]) * N_best for i in range(len(W_index))]
if (tyukan == 1):
    for w in range(len(WP_list)):
        Otb_B = [ Otb_B[i] + int(W_index[i] == Goal_Word_albert[int(WP_list[w])]) * N_best for i in range(len(Otb_B)) ]

print("BoW:", Otb_B)

#Path-Planning
#Path, Path_ROS, PathWeightMap, Path_one = PathPlanner(Otb_B, Start_Position[int(init_position_num)], THETA, CostMapProb) #gridmap, costmap)

S_Nbest = Otb_B
#THETAを展開
W, W_index, Mu, Sig, Pi, Phi_l, K, L = THETA

#ROSの座標系の現在位置を2-dimension array index にする
X_init_index = start_inv ###TEST  #Map_coordinates_To_Array_index(X_init)
print("Initial Xt:",X_init_index)

#length and width of the MAP cells
map_length = len(CostMapProb)     #len(costmap)
map_width  = len(CostMapProb[0])  #len(costmap[0])
print("MAP[length][width]:",map_length,map_width)

#Pre-calculation できるものはしておく
LookupTable_ProbCt = np.array([multinomial.pmf(S_Nbest, sum(S_Nbest), W[c])*Pi[c] for c in range(L)])  #Ctごとの確率分布 p(St|W_Ct)×p(Ct|Pi) の確率値
###SaveLookupTable(LookupTable_ProbCt, outputfile)
###LookupTable_ProbCt = ReadLookupTable(outputfile)  #Read the result from the Pre-calculation file(計算する場合と大差ないかも)


print("Please wait for PostProbMap")
output = outputfile + "N"+str(N_best)+"G"+str(speech_num) + "_PathWeightMap.csv"
#if ITO == 1:
#    PathWeightMap = PostProbMap_nparray_jit_ITO(CostMapProb,Mu,Sig,Phi_l,LookupTable_ProbCt,map_length,map_width,L,K) #,IndexMap)  # Ito
#
#    #[TEST]計算結果を先に保存
#    save_data.SaveProbMap(PathWeightMap, outputfile, speech_num)
#else:
if (os.path.isfile(output) == False): # or (UPDATE_PostProbMap == 1):  #すでにファイルがあれば作成しない
    #PathWeightMap = PostProbMap_jit(CostMapProb,Mu,Sig,Phi_l,LookupTable_ProbCt,map_length,map_width,L,K) #マルチCPUで高速化できるかも #CostMapProb * PostProbMap #後の処理のために, この時点ではlogにしない
    start_PWM_time = time.time()
    PathWeightMap = path_calculate.PostProbMap_nparray_jit(CostMapProb,Mu,Sig,Phi_l,LookupTable_ProbCt,map_length,map_width,L,K) #,IndexMap) 
    end_PWM_time = time.time()
    if (SAVE_time == 1):
        time_pp = end_PWM_time - start_PWM_time #end_recog_time
        fp = open( outputfile + "N"+str(N_best)+"G"+str(speech_num) + "_time_PathWeightMap.txt", 'w')
        fp.write(str(time_pp)+"\n")
        fp.close()

    #[TEST]計算結果を先に保存
    save_data.SaveProbMap(PathWeightMap, outputfile, speech_num)
else:
    PathWeightMap = read_data.ReadProbMap(outputfile, speech_num)
    #print("already exists:", output)

#Read the emission probability file 
#PathWeightMap = read_data.ReadProbMap(outputfile, speech_num)

#CostMapProb = ReadCostMapProb(outputfile) #Ito
#path_calculate.PathPlanner(path_calculate, N_best, 1, THETA, CostMapProb, outputfile, speech_num, outputname) #Ito


#####描画
plt.imshow(map_image,cmap='gray')
#plt.imshow(gridmap + (40+1)*(gridmap == -1), origin='lower', cmap='binary', vmin = 0, vmax = 100, interpolation='none') #, vmin = 0.0, vmax = 1.0)
#plt.xticks(rotation=90)
plt.tick_params(axis='x', which='major', labelsize=8)
plt.tick_params(axis='y', which='major', labelsize=8)
plt.xlim([380,800])             #x軸の範囲
plt.ylim([180,510])             #y軸の範囲
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
J = Sampling_J #len(goal_candidate)
if(J != THETA[6]):
    print("[WARNING] J is not K",J,K)
p_cost_candidate = [0.0 for j in range(J)]
Path_candidate = [[0.0] for j in range(J)]

###goal候補ごとにA*を実行
for gc_index in range(J):
    goal = goal_candidate[gc_index]
    Path, p_cost = a_star(start_inv, goal, gridmap, action_functions, cost_of_actions, PathWeightMap)    
    p_cost_candidate[gc_index] = p_cost #/ float(len(Path))
    Path_candidate[gc_index] = Path


### select the goal of minimum cost
min_gc_index = np.argmin(p_cost_candidate)
Path = Path_candidate[min_gc_index]
goal = goal_candidate[min_gc_index]
print("Goal:", goal)


if (SAVE_time == 1):
    #PP終了時刻を保持
    end_pp_time = time.time()
    time_pp = end_pp_time - start_time #end_recog_time
    fp = open( outputname + "_time_pp.txt", 'w')
    fp.write(str(time_pp)+"\n")
    fp.close()
    
for i in range(len(Path)):
  plt.plot(Path[i][0], Path[i][1], "s", color="red", markersize=1)


#The moving distance of the path
Distance = tools.PathDistance(Path)

#Save the moving distance of the path
save_data.SavePathDistance(Distance, outputname)
print("Path distance using A* algorithm is "+ str(Distance))

#計算上パスのx,yが逆になっているので直す
Path_inv = [[Path[t][1], Path[t][0]] for t in range(len(Path))]
Path_inv.reverse()
Path_ROS = Path_inv #使わないので暫定的な措置
#Save the path
save_data.SavePath(start_inv, [goal[1], goal[0]], Path_inv, Path_ROS, outputname)


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
save_data.SaveLogLikelihood(LogLikelihood_step,0,0, outputname)

#すべてのステップにおける累積報酬（sum log likelihood）の値を保存
save_data.SaveLogLikelihood(LogLikelihood_sum,1,0, outputname)


#plt.show()

#Save path trajectory in the map as a color image
#output = outputfile + "N"+str(N_best)+"G"+str(speech_num)
plt.savefig(outputname + '_Path.png', dpi=300)#, transparent=True
#plt.savefig(outputfile + "step/" + conditions + '_Path_Weight' +  str(temp).zfill(3) + '.png', dpi=300)#, transparent=True
plt.savefig(outputname + '_Path.pdf', dpi=300)#, transparent=True
plt.clf()

print("[END] SpCoNavi. (A star)")