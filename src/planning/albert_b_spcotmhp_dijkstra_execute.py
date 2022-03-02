#!/usr/bin/env python
#coding:utf-8

import os.path
import sys
import time
from scipy import append
#from math import log
from scipy.stats import multivariate_normal,multinomial
import matplotlib.pyplot as plt
from PIL import Image,ImageOps #, ImageDraw
import spconavi_read_data
import spconavi_save_data
from __init__ import *
from submodules import *

##Command: 
#python3 albert_b_spconavi_astar_execute.py trialname mapname iteration sample init_position_num speech_num initial_position_x initial_position_y waypoint_word
#python3 albert_b_spconavi_astar_execute.py albertGMM01 map 1 9 init 0 590 340 -1
#python3 albert_b_spconavi_astar_execute.py albertGMM01 map 1 9 init 1 650 380 10

#python3 albert_b_spcotmhp_dijkstra_execute.py trialname iteration sample init_position_num speech_num initial_position_x initial_position_y waypoint_word
#python3 albert_b_spcotmhp_dijkstra_execute.py albertL2R01 1 2 -1 1 500 250 0
#python3 albert_b_spcotmhp_dijkstra_execute.py albertTMHP01 1 6 -1 1 500 250 0

#python3 albert_b_spcotmhp_dijkstra_execute.py albertTMHP01 1 6 -1 1 650 380 10
#python3 albert_b_spcotmhp_dijkstra_execute.py albertL2R01 1 2 -1 1 650 380 10

tools     = spconavi_read_data.Tools()
read_data = spconavi_read_data.ReadingData()
save_data = spconavi_save_data.SavingData()


#Definition of action (functions in spconavi_read_data)
action_functions = [tools.right, tools.left, tools.up, tools.down, tools.stay] #, migiue, hidariue, migisita, hidarisita]
cost_of_actions  = np.log( np.ones(len(action_functions)) / float(len(action_functions)) ) #[    1/5,    1/5,  1/5,    1/5,    1/5]) #, ,    1,        1,        1,          1]

K=20
L=20


#GaussMap make (same to the function in spcotmhp_viterbi_path_calculate) #Ito
def PostProbMap_Gauss(CostMapProb,Mu,Sig,map_length,map_width,it): #,IndexMap):
        origin_pos = tools.Array_index_To_Map_coordinates_albert([0.0,0.0])
        end_pos    = tools.Array_index_To_Map_coordinates_albert([map_width-1,map_length-1])
        x,y = np.meshgrid(np.linspace(origin_pos[0],end_pos[0],map_width),np.linspace(origin_pos[1],end_pos[1],map_length))  #np.meshgrid(np.linspace(-10.0,9.92,map_width),np.linspace(-10.0,9.92,map_length))
       
        pos = np.dstack((x,y))    
        #PostProbMap = np.array([ [ PostProb_ij([width, length],Mu,Sig,map_length,map_width, CostMapProb,it) for width in xrange(map_width) ] for length in xrange(map_length) ])
        PostProb=multivariate_normal(Mu[it],Sig[it]).pdf(pos)

        return CostMapProb * PostProb


# The weight of the topoligical-level likelihood in word Se
def TopoWeight(i,j,It,w,THETA,psi):
    #THETAを展開
    W, W_index, Myu, S, pi, phi, K, L = THETA
    ignore_value = 1.0 / K
    if (psi[i][j] >= ignore_value) and (list(It).count(i) > 1) and (list(It).count(j) > 1):
        # i_e: i, i_e-1: j, S_e:w
        sum_c  = np.sum([phi[c][i]*(W[c][w]**N_best)*pi[c] for c in range(L)])
        bunbo2 = np.sum([phi[c][i] for c in range(L)])
        weight = psi[i][j] * sum_c / bunbo2
    else:
      weight = 1.0

    return weight



###↓### Sampling of goal candidates ############################################
def EstimateGoal(Otb_B, THETA):
    #THETAを展開
    W, W_index, Mu, S, pi, phi_l, K, L = THETA
    
    #Prob math func of p(it | Otb_B, THETA) = Σc p(it | phi_c)p(st=Otb_B | Wc)p(c | pi)
    pmf_it = np.ones(K)
    for k in range(K):
        pmf_it[k] = np.sum([ phi_l[c][k] * multinomial.pmf(Otb_B, sum(Otb_B), W[c]) * pi[c] for c in range(L) ])
        bunbo2 = np.sum([phi_l[c][k] for c in range(L)])
        pmf_it[k] = pmf_it[k] / bunbo2

    #argmax_it = np.argmax(pmf_it)

    
    #Normalization
    pmf_it_n = np.array([pmf_it[i] / float(np.sum(pmf_it)) for i in range(K)])

    #bests_five  = []
    upper_zero_three = np.array([1.0 for i in range(K)])
    for i in range(K):
        if (pmf_it_n[i] < 0.3):
            upper_zero_three[i] = 0.0

    pmf_it = pmf_it * upper_zero_three
    #Normalization2
    pmf_it_n = np.array([pmf_it[i] / float(np.sum(pmf_it)) for i in range(K)])

    #Sampling it from multinomial distribution
    sample_it = np.random.choice(K, size=Sampling_G, replace=True, p=pmf_it_n)
    #random.choices([i for i in range(K)], k=Sampling_G, weights=pmf_it_n)
    #multinomial.rvs(Sampling_G, pmf_it_n, size=1, random_state=None)
    #print(sample_it)
    
    return sample_it

    """
    goal_candidate = []
    for it in range(K):
        count_it = 0
        while (count_it < sample_it[0][it]):
            goal_candidate += [tools.Map_coordinates_To_Array_index_albert(multivariate_normal.rvs(mean=Myu[it], cov=S[it], size=1, random_state=None))]
            count_it += 1
    #Xt_max = Map_coordinates_To_Array_index_albert( [ Xp[pox.index(max(pox))][0], Xp[pox.index(max(pox))][1] ] ) #[0.0,0.0] ##確率最大の座標候補
    goal_candidate_tuple = [(goal_candidate[j][1], goal_candidate[j][0]) for j in range(Sampling_G)]
    print("Goal candidates:", goal_candidate_tuple)
    return goal_candidate_tuple
    """
 
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
            q_pev = tools.Manhattan_distance(q, goal) * np.log(float(len(action_functions))) #*10.0 #heuristic function
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



    
def ReadDistanceMtrx(outputfile):
    dis  = [ [0.0 for atem in range(K)] for aky in range(K) ]
    c=0
    for line in open(outputfile + 'Astar_SpCoTMHP_distance.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in range(len(itemList)):
            if itemList[i] != "":
              dis[c][i] = float(itemList[i])
        c = c + 1
    return dis


def ReadCostMtrx(outputfile):
    cost  = [ [0.0 for atem in range(K)] for aky in range(K) ]
    c=0
    for line in open(outputfile + 'Astar_SpCoTMHP_cost.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in range(len(itemList)):
            if itemList[i] != "":
              cost[c][i] = float(itemList[i])
        c = c + 1
    return cost
        

def ReadPathDistance(outputsubfolder,st,gl):
    for line in open(outputsubfolder + "Astar_SpCoTMHP_S" + str(st) + "_G" + str(gl) + "_Distance.csv", 'r'):
        distance = float(line[:-1])

    return int(distance)

def ReadLogLikelihood(outputsubfolder,st,gl):
    LogLikelihood  = [ 0.0 ]
    c=0
    for line in open(outputsubfolder + "Astar_SpCoTMHP_S" + str(st) + "_G" + str(gl) + "_Log_likelihood_sum.csv", 'r'):
        #itemList = line[:-1].split(',')
        #for i in range(len(itemList)):
        #    if itemList[i] != "":
        LogLikelihood.append(float(line[:-1]))
        c = c + 1
    dist = int(ReadPathDistance(outputsubfolder,st,gl))
    #print(st,gl,dist,len(LogLikelihood)) #,LogLikelihood)
    if (dist > T_horizon):
        dist = T_horizon
    #print(LogLikelihood[dist])

    return LogLikelihood[dist]

def ReadLogLkhdMtrx(outputsubfolder,psi,It,K):
    ignore_value = 1.0 / K
    LogLkhdMtrx  = np.array([ [ 0.0 for gl in range(K)] for st in range(K) ])
    for st in range(K):
        for gl in range(K):
            if  (psi[st][gl] >= ignore_value) and (list(It).count(st) > 1) and (list(It).count(gl) > 1) and (st != gl):
                LogLkhdMtrx[st][gl] = ReadLogLikelihood(outputsubfolder,st,gl)
    return LogLkhdMtrx

        
# おそらく最適探索（Dijkstra）が実装されている
def SearchTopoDijkstra(glf, st, cost): # goal index, start index, cost
    ev   = [ 0.0 for aky in range(K) ]
    node = [ [0,-2] for aky in range(K) ]
    node[st]=[1,-1]

    fin=0
    while fin != 1:
      for i in range(K):
        if node[i][0]==2:
            for k in range(K):  
                if cost[i][k]>0:
                    if node[k][0]==0:
                        node[k][0]=1
                        node[k][1]=i
                        ev[k]=ev[i]+cost[i][k]
                    elif node[k][0]==1:
                        if ev[k]>ev[i]+cost[i][k]:
                            ev[k]=ev[i]+cost[i][k]
                            node[k][1]=i
            node[i][0]=3
            break
           
      minm=100000000000000000  
      minm_i=-1   
      for i in range(K):
        if node[i][0]==1:
          if minm > ev[i]:
            minm=ev[i]  
            minm_i=i
      if minm_i == glf:          
         fin=1
         #print("fin")
      else:
       node[minm_i][0]=2
    
    i=glf
    root=[]    
    while i!= -1 :
       for k in range(K):
         if i==k:
           root.insert(0,k)
           i=node[k][1]
    return root


# おそらく動的計画法（Viterbi）が実装されている
def SearchTopoViterbi(s_i, lk, Step): # start index, likelihoods, T_topo
    lkt=[ [0.0 for i in range(K)] for k in range(Step)]#sum likewood
    rt=[ [-1 for i in range(K)] for k in range(Step)]#oya node save
    node =[0 for i in range(K) ] #
    node[s_i]=1
    ndc=[0 for i in range(K) ]#
    root=[0 for i in range(Step) ]
    mxs=-1#
    mxg=-1
    mx=0
    #print(lk)
    #print(node)

    for t in range(Step):
      for i in range(K):
        if t==0:
          if node[i]==1:
            for j in range(K):
              if lk[i][j]>0:
                 lkt[t][j]=lk[i][j]
                 rt[t][j]=i
                 ndc[j]=1
        else:
          if node[i]==1:
            for j in range(K):
              if lk[i][j]>0:
                if lkt[t][j]<lkt[t-1][i]+lk[i][j]:
                  lkt[t][j]=lkt[t-1][i]+lk[i][j]
                  rt[t][j]=i
                  ndc[j]=1
      
      for f in range(K):
         if ndc[f]==1 and node[f]==0:
           node[f]=1
           
    #print(rt)        
    for i in range(K):
        if mx<lkt[Step-1][i]:
          mx=lkt[Step-1][i]
          mxg=i
    #print(mxg)                  
    
    # Backword path              
    for i in range(Step):   
        root[Step-i-1]=rt[Step-i-1][mxg]
        mxg=rt[Step-i-1][mxg]

    return root


## パスを読み込み、つなげる (For A*)
def PathConnect(outputsubfolder, dis, Path_ROS, root):
    for v in range(len(root)-1):
        path=[ (0,0) for k in range(int(dis[root[v]][root[v+1]]))]
        #print(dis[root[v]][root[v+1]])
        i=0
        if ( root[v] != root[v+1] ):
            output = outputsubfolder + "Astar_SpCoTMHP_S" + str(root[v]) + '_G' + str(root[v+1]) +  '_Path_ROS.csv'
            if (os.path.exists(output)):
                output = outputsubfolder + "Astar_SpCoTMHP_S" + str(root[v]) + '_G' + str(root[v+1]) +  '_Path_ROS.csv'
            else:
                output = outputsubfolder + "Astar_SpCoTMHP_S" + str(root[v+1]) + '_G' + str(root[v]) +  '_Path_ROS.csv'
                path=[ (0,0) for k in range(int(dis[root[v+1]][root[v]]))]
            print(output)
            for line in open(output, 'r'):
                itemList = line[:-1].split(',')
                path[i] = ( float(itemList[0]) , float(itemList[1]) )
                #path[i][0]=int(po[0])
                #path[i][1]=int(po[1])
                i = i + 1
                #print(i)
            Path_ROS=Path_ROS+path


    return Path_ROS


#################################################
if __name__ == '__main__': 
    print("[START] SpCoTMHP (Dijkstra).")

    #Request a folder name for learned parameters.
    trialname = sys.argv[1]

    #Request iteration value
    iteration = sys.argv[2] #1

    #Request sample value
    sample = sys.argv[3] #0

    #Request the index number of the robot initial position
    init_position_num = int(sys.argv[4]) #位置分布インデックス(-1で座標直接指定)

    #Request the file number of the speech instruction   
    speech_num = sys.argv[5] #0
    
    #初期値を指定する場合はTHETA読み込み以降に記載(int(sys.argv[6]) , int(sys.argv[7]))

    #中間地点の単語番号を指定 (未実装：複数指定の場合、コンマ区切りする)
    waypoint_word = sys.argv[8] # -1:中間なし


    WP_list    = waypoint_word[:].split(',')
    print("WP:", WP_list)

    if (int(WP_list[0]) == -1):
        tyukan = 0
    else:
        tyukan = 1


    ##FullPath of folder
    filename = outputfolder_albert + trialname #+ "/" 
    print(filename, iteration, sample)
    outputfile      = filename + navigation_folder #outputfolder + trialname + navigation_folder
    outputsubfolder = outputfile + "astar_node_gauss/"
    outputname_d    = outputfile + "dijkstra_result_wd/"

    Makedir( outputname_d )

    #Read the files of learned parameters  #THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
    THETA   = read_data.ReadParameters(iteration, sample, filename, trialname)
    W, W_index, Mu, Sig, pi, phi_l, K, L = THETA
    psi   = read_data.ReadPsi(iteration, sample, filename, trialname)
    

    gridmap = read_data.ReadMap(outputfile)
    map_length, map_width = gridmap.shape
    
    CostMapProb = read_data.ReadCostMapProb(outputfile)
    #CostMap = read_data.ReadCostMap(outputfile)
    #CostMapProb = (100.0 - CostMap)/100


    # map の .pgm file を読み込み
    map_file_path = inputfolder_albert + map_file + '.pgm' #roslib.packages.get_pkg_dir('em_spco_ae') + '/map/' + self.map_file + '/map.pgm'
    map_image     = Image.open(map_file_path)
    map_image     = ImageOps.flip(map_image)      # 上下反転

    # height and width を得る
    #width, height = map_image.size
    #print(map_image.size)


    #描画
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

 
    

    # 初期位置の設定
    if (init_position_num == -1): #初期値を指定する場合
        start = [int(sys.argv[6]) , int(sys.argv[7])]
        start_inv = [start[1], start[0]]
        print("Start:", start)
    else: #初期値は__init__.pyのリストから選択する場合
        start_inv = Start_Position[init_position_num]
        start = [start_inv[1], start_inv[0]]
    

    outputname      = outputname_d + "Dijkstra_SpCoTMHP_"+"T"+str(T_topo)+"S"+str(start)+"H"+str(waypoint_word)+"G"+str(speech_num)

    dis  = ReadDistanceMtrx(outputfile)      
    cost = ReadCostMtrx(outputfile)    
    #lk   = -1.0 * np.array(cost)  # cost is -log likelihood  

    psi     = read_data.ReadPsi(iteration, sample, filename, trialname) #[ [0.0 for atem in range(K)] for aky in range(K) ]
    It     = read_data.ReadIt(iteration, sample, filename, trialname)


    #Read CoonectMatricx (psi_setting.csv) file
    #CoonectMatricx=[ [0.0 for i in range(K)] for c in range(K) ]
    #c = 0
    #for line in open(filename + "/" + trialname + '_psi_setting.csv', 'r'):
    #    itemList = line[:-1].split(',')
    #    for i in range(len(itemList)):
    #        if itemList[i] != "":
    #          CoonectMatricx[c][i] = float(itemList[i])
    #    c = c + 1

    lk = ReadLogLkhdMtrx(outputsubfolder,psi,It,K) # log likelihood

    ### 事前計算として、W_indexごとにトポロジカル重みを計算
    TopoWeight_W_index = [ np.log(np.array([ [TopoWeight(i,j,It,w,THETA,psi) for i in range(K)] for j in range(K) ])) for w in range(len(W_index)) ]


    same_lk = np.array([multivariate_normal(Mu[it],Sig[it]).pdf(Mu[it])*0.5 for it in range(K)])
    #same_lk = np.array([CostMapProb[tools.Map_coordinates_To_Array_index_albert(Mu[it])[0]][tools.Map_coordinates_To_Array_index_albert(Mu[it])[1]] for it in range(K)])
    same_lk = np.diag(np.log(same_lk))
    #print("same_lk",same_lk)

    duration_term_lk = np.array([ [-1.0 * dis[i][j] * np.log(T_topo) for j in range(K)] for i in range(K) ]) # log duration prob


    whole_log_lk_inv_W_index =  -1.0 * np.array([TopoWeight_W_index[w] + lk + same_lk + duration_term_lk for w in range(len(W_index))])

    #weight_inv = -1.0 * (TopoWeight_W_index[S_list[t]] + lk + np.diag(same_lk))
    weight_W_index = whole_log_lk_inv_W_index #[ whole_log_lk_inv_W_index[w] - np.min(whole_log_lk_inv_W_index[w]) for w in range(len(W_index))]
    #[np.max(whole_log_lk_inv_W_index[w]) - whole_log_lk_inv_W_index[w] for w in range(len(W_index))]


    if (SAVE_time == 1):
      #Substitution of start time
      start_time = time.time()

    ### 初期位置から一番近くの位置分布を推定（ガウスの重みが一番高い分布のインデックス）
    near_node = np.argmax([multivariate_normal(Mu[k],Sig[k]).pdf(tools.Array_index_To_Map_coordinates_albert(start)) for k in range(K)])
        
        
    print("near_node",near_node,"start A*")
    #ss=Map_coordinates_To_Array_index_albert(Mu[S_i])
    #start=(150,130)

    ###v###近くの位置分布のindexのガウスの平均をゴールとしたA*を実行###v###
    #PathWeightMap = PostProbMap_Gauss(CostMapProb,Mu,Sig,map_length,map_width,near_node)

    gg=tools.Map_coordinates_To_Array_index_albert(Mu[near_node])
    #print(gg[0])
    goal_candidate=[[gg[1],gg[0]]]
    #goal_candidate=[[55,135],[89,103]]
    #J = len(goal_candidate)
    J=1
    #if(J != THETA[6]):
    # print("[WARNING] J is not K",J,K)
    #p_cost_candidate = [0.0 for j in range(J)]
    Path_candidate   = [[0.0] for j in range(J)]
    #print(goal_candidate)

    ###goal候補ごとにA*を実行
    for gc_index in range(J):
        goal = goal_candidate[gc_index]
        #Path, p_cost = a_star(start_inv, goal, gridmap, action_functions, cost_of_actions, CostMapProb)    
        temp = start_inv
        #for susumu in range(Manhattan_distance(start_inv,goal)):
        susumu = 0
        Path = [(start_inv[0],start_inv[1])]
        while (Path[-1][0] != goal[0]) and (Path[-1][1] != goal[1]):
                if (susumu % 2):
                    if (Path[-1][0] > goal[0]):
                        #temp =  (temp[0]-1, temp[1]) 
                        Path = Path + [ (temp[0]-1, temp[1]) ]
                    else:
                        Path = Path + [ (temp[0]+1, temp[1]) ]
                else:
                    if (Path[-1][1] > goal[1]):
                        Path = Path + [ (temp[0], temp[1]-1) ]
                    else:
                        Path = Path + [ (temp[0], temp[1]+1) ]
                temp = Path[-1]
                susumu += 1
                print(susumu,Path)
        #Path = [start_inv, goal]

        #first_cost = p_cost / float(len(Path))
        #p_cost_candidate[gc_index] = p_cost / float(len(Path)+1)
        Path_candidate[gc_index] = Path    


    ### select the goal of expected cost
    expect_gc_index = 0 #np.argmin(p_cost_candidate)
    Path = Path_candidate[expect_gc_index]
    goal = goal_candidate[expect_gc_index]
    print("Goal:", goal)



    #計算上パスのx,yが逆になっているので直す
    Path_inv = [ (Path[t][1], Path[t][0]) for t in range(len(Path)) ]
    Path_inv.reverse()
    Path_ROS = Path_inv #使わないので暫定的な措置
    #パスを保存
    #SavePath(start, [goal[1], goal[0]], Path_inv, Path_ROS, outputname)

    #The moving distance of the path
    Distance = tools.PathDistance(Path_inv)
    print("Path distance using A* algorithm is "+ str(Distance))

    ###^###近くの位置分布のindexのガウスの平均をゴールとしたA*を実行###^###





    #####Estimate the goal point by spatial concept
    Otb_B      = [int(W_index[i] == Goal_Word_albert[int(speech_num)]) * N_best for i in range(len(W_index))]   
    print("BoW:", Otb_B)

    Otb_B_list = [ [int(W_index[i] == Goal_Word_albert[int(WP_list[w])]) * N_best for i in range(len(W_index))] for w in range(len(WP_list))]
    print("WP:BoW_list:", Otb_B_list)
    len_Otb_B_list = len(Otb_B_list)

    #st = near_node
    gl_J_best = [0.0 for j in range(Sampling_G)]
    root_list = [ [] for j in range(Sampling_G)]
    Path_list = [Path_ROS for j in range(Sampling_G)]
    Path_log_likelihood_list = [0.0 for j in range(Sampling_G)]
    S_list = [] # W_indexの番号の配列
    #i = 0
  
    ## 最終ゴール（位置分布インデックス）の候補リストを推定
    gl_J_best = EstimateGoal(Otb_B, THETA)
    print("gl:",gl_J_best)

    for j in range(Sampling_G):
        i=0
        #tyukan = 0
        st = near_node
        if (tyukan == 1):
            for i in range(len(Otb_B_list)):
                ### 中間地点が指定されている場合、中間地点（がなくなる）までパスプランニング、中間地点がなければゴールへのプランニング ###
                #for j_tyukan in range(Sampling_G):
                #print("tyukan",i)
                ## 次のゴール（位置分布インデックス）を推定
                tyukan_J_best = EstimateGoal(Otb_B_list[i], THETA)
                print("st,tyukan:",st,tyukan_J_best[j])

                ## 次のゴールまでのトポロジカルプランニング
                root = SearchTopoDijkstra(tyukan_J_best[j], st, weight_W_index[np.argmax(Otb_B_list[i])])
                root_list[j] = root_list[j] + root
                print("root_t:",j,root_list[j])

                ## パスの尤度を計算し保存
                for r in range(len(root)):
                    if r == 0:
                        Path_log_likelihood_list[j] += weight_W_index[np.argmax(Otb_B_list[i])][st][root[r]]
                    else:
                        Path_log_likelihood_list[j] += weight_W_index[np.argmax(Otb_B_list[i])][root[r-1]][root[r]]

                ## パスを読み込み、つなげる
                Path_list[j] = PathConnect(outputsubfolder, dis, Path_list[j], root) 

                root = root_list[j]
                ## T_topo/(len_Otb_B_list+1))まで最終地点の値を埋める
                for _ in range(int(T_topo/(len_Otb_B_list+1) - len(root))):
                    Path_log_likelihood_list[j] += weight_W_index[np.argmax(Otb_B)][root[-1]][root[-1]]

                st = tyukan_J_best[j]
                #i = i + 1

            ## 中間地点がなければゴールへのプランニングへ
            #if (i >= len(Otb_B_list)):
            #    tyukan = 0


        ## 次のゴールまでのトポロジカルプランニング
        root = SearchTopoDijkstra(gl_J_best[j], st, weight_W_index[np.argmax(Otb_B)])
        root_list[j] = root_list[j] + root
        print("root_j:",j,root_list[j])


        ## パスを読み込み、つなげる
        Path_list[j] = PathConnect(outputsubfolder, dis, Path_list[j], root) 

        root = root_list[j]
        ## パスの尤度を計算し保存
        for r in range(len(root)):
            if r == 0:
                Path_log_likelihood_list[j] += weight_W_index[np.argmax(Otb_B)][st][root[r]]
            else:
                Path_log_likelihood_list[j] += weight_W_index[np.argmax(Otb_B)][root[r-1]][root[r]]

        ## T_topoまで最終地点の値を埋める
        for _ in range(T_topo - len(root)):
            Path_log_likelihood_list[j] += weight_W_index[np.argmax(Otb_B)][root[-1]][root[-1]]


    ## 候補のパスから尤度が最大(-log likelihoodなので最小)のものを選ぶ
    saiyou = np.argmin(Path_log_likelihood_list)
    print("saiyou",saiyou,root_list[saiyou])
    Path_ROS = Path_list[saiyou]



    if (SAVE_time == 1):
        #PP終了時刻を保持
        end_pp_time = time.time()
        time_pp = end_pp_time - start_time #end_recog_time
        fp = open( outputname + "_time_pp.txt", 'w')
        fp.write(str(time_pp)+"\n")
        fp.close()

    print("[END] SpCoTMHP (Dijkstra).")
    # print(Path_ROS)

    disp = [tools.PathDistance(Path_ROS)] # tupple in list  #len(Path_ROS)

    for i in range(len(Path_ROS)):
      plt.plot(Path_ROS[i][1], Path_ROS[i][0], "s", color="red", markersize=1) #tab:red
    plt.savefig(outputname + '_Path.pdf', dpi=300)#, transparent=True
    plt.savefig(outputname + '_Path.png', dpi=300)#, transparent=True
    
    np.savetxt(outputname + "_fin_Path_ROS.csv", Path_ROS, delimiter=",")
    np.savetxt(outputname + "_fin_Distance.csv", disp, delimiter=",")
    plt.clf()
    print("DONE.",outputname)
